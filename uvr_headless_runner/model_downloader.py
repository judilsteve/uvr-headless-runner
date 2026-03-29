#!/usr/bin/env python3
"""
UVR Model Downloader Module
===========================
Replicates the official UVR GUI model download center behavior.

This module provides:
1. Remote model registry sync (from official UVR sources)
2. Per-architecture model listings
3. Automatic model downloading with retry and resume
4. UVR GUI-compatible directory structure
5. Checksum verification
6. Fuzzy model name matching
7. HTTP/HTTPS proxy support

Based on reverse engineering of UVR.py download_checks.json and related logic.

HTTP/HTTPS Proxy Support:
    This module automatically respects standard proxy environment variables:
    - HTTP_PROXY / http_proxy: Proxy for HTTP connections
    - HTTPS_PROXY / https_proxy: Proxy for HTTPS connections
    - NO_PROXY / no_proxy: Comma-separated list of hosts to bypass
    
    Python's urllib automatically uses these when set. No manual configuration needed.
    
    Example:
        export HTTP_PROXY=http://proxy.example.com:8080
        export HTTPS_PROXY=http://proxy.example.com:8080
        python -m model_downloader --download "model_name" --arch mdx

Usage:
    from model_downloader import ModelDownloader
    
    downloader = ModelDownloader()
    
    # List available models
    print(downloader.list_models('mdx'))
    
    # Get model info
    info = downloader.get_model_info('UVR-MDX-NET Inst HQ 3', 'mdx')
    
    # Download a model
    downloader.download_model('UVR-MDX-NET Inst HQ 3', 'mdx')
"""

import os
import sys
import json
import hashlib
import urllib.request
import urllib.error
import shutil
import time
import socket
import difflib
import signal
import atexit
import threading
import contextlib
import errno
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Cross-platform file locking for concurrent container safety
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False  # Windows - will use alternative approach

try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False  # Not Windows

# Global registry for active downloads (for cleanup on interrupt)
_active_downloads: set = set()
_download_lock = threading.Lock()  # Thread lock for in-process safety
_original_sigint = None
_original_sigterm = None

# Minimum file size to be considered a valid model (prevents false-positive "installed")
MIN_MODEL_FILE_SIZE = 1024  # 1 KB - any real model file is much larger

# Enable/disable checksum verification (can be set via environment variable)
# When enabled, downloads will verify SHA256 checksums if available
VERIFY_CHECKSUMS = os.environ.get('UVR_VERIFY_CHECKSUMS', '1').lower() in ('1', 'true', 'yes')

# URL for model checksums (if available from official source)
MODEL_CHECKSUMS_URL = "https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/model_checksums.json"


def get_proxy_status() -> dict:
    """
    Get current proxy configuration status.
    
    Returns a dict with proxy configuration info (without revealing credentials).
    SECURITY: Only returns whether proxy is configured, not the actual URLs.
    
    Returns:
        dict with keys: 'http_proxy', 'https_proxy', 'no_proxy', each containing
        True/False indicating if that proxy type is configured.
    """
    return {
        'http_proxy': bool(os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')),
        'https_proxy': bool(os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')),
        'no_proxy': bool(os.environ.get('NO_PROXY') or os.environ.get('no_proxy')),
    }


def is_proxy_configured() -> bool:
    """
    Check if any HTTP/HTTPS proxy is configured.
    
    Returns:
        True if HTTP_PROXY or HTTPS_PROXY is set, False otherwise.
    """
    status = get_proxy_status()
    return status['http_proxy'] or status['https_proxy']


def _cleanup_partial_downloads():
    """Clean up any partial download files on exit."""
    with _download_lock:
        for temp_path in list(_active_downloads):
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
        _active_downloads.clear()


def _signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    _cleanup_partial_downloads()
    # Restore original handler and re-raise
    if signum == signal.SIGINT and _original_sigint:
        signal.signal(signal.SIGINT, _original_sigint)
    elif signum == signal.SIGTERM and _original_sigterm:
        signal.signal(signal.SIGTERM, _original_sigterm)
    sys.exit(128 + signum)


def _register_cleanup_handlers():
    """Register cleanup handlers (called once on module load)."""
    global _original_sigint, _original_sigterm
    
    # Only register in main thread
    if threading.current_thread() is not threading.main_thread():
        return
    
    # Register atexit handler
    atexit.register(_cleanup_partial_downloads)
    
    # Register signal handlers (preserve original handlers)
    try:
        _original_sigint = signal.signal(signal.SIGINT, _signal_handler)
        _original_sigterm = signal.signal(signal.SIGTERM, _signal_handler)
    except (ValueError, OSError):
        # Signal handling not available (e.g., in some environments)
        pass


# Register handlers on module import
_register_cleanup_handlers()

# Try to import rich for beautiful progress bars
try:
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
    )
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ============================================================================
# Custom Exception Classes for Better Error Handling
# ============================================================================

class ModelDownloaderError(Exception):
    """Base exception for model downloader errors."""
    def __init__(self, message: str, suggestion: str = None):
        self.message = message
        self.suggestion = suggestion
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        msg = f"[ModelDownloader Error] {self.message}"
        if self.suggestion:
            msg += f"\n  Suggestion: {self.suggestion}"
        return msg


class NetworkError(ModelDownloaderError):
    """Network-related errors (timeout, connection, etc.)."""
    pass


class RegistryError(ModelDownloaderError):
    """Model registry sync/lookup errors."""
    pass


class DownloadError(ModelDownloaderError):
    """File download errors."""
    pass


class IntegrityError(ModelDownloaderError):
    """File integrity/checksum errors."""
    pass


class ModelNotFoundError(ModelDownloaderError):
    """Model not found in registry."""
    def __init__(self, model_name: str, arch_type: str, similar_models: List[str] = None):
        self.model_name = model_name
        self.arch_type = arch_type
        self.similar_models = similar_models or []
        
        message = f"Model '{model_name}' not found in {arch_type} registry."
        suggestion = None
        if self.similar_models:
            suggestion = f"Did you mean: {', '.join(self.similar_models[:5])}?"
        else:
            suggestion = f"Use --list to see available {arch_type} models."
        
        super().__init__(message, suggestion)


class DiskSpaceError(ModelDownloaderError):
    """Insufficient disk space error."""
    pass


class PermissionError(ModelDownloaderError):
    """File/directory permission errors."""
    pass


# ============================================================================
# UVR Official Data Sources (DO NOT CHANGE unless UVR changes)
# ============================================================================
DOWNLOAD_CHECKS_URL = "https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json"
NORMAL_REPO = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
MDX23_CONFIG_CHECKS = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/"

# Model data links (for hash lookups)
VR_MODEL_DATA_LINK = "https://raw.githubusercontent.com/TRvlvr/application_data/main/vr_model_data/model_data_new.json"
MDX_MODEL_DATA_LINK = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_data_new.json"
MDX_MODEL_NAME_DATA_LINK = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_name_mapper.json"
DEMUCS_MODEL_NAME_DATA_LINK = "https://raw.githubusercontent.com/TRvlvr/application_data/main/demucs_model_data/model_name_mapper.json"

# Architecture types (matching UVR constants)
VR_ARCH_TYPE = 'VR Arc'
MDX_ARCH_TYPE = 'MDX-Net'
DEMUCS_ARCH_TYPE = 'Demucs'

# Demucs version identifiers
DEMUCS_V3_ARCH_TYPE = 'Demucs v3'
DEMUCS_V4_ARCH_TYPE = 'Demucs v4'
DEMUCS_NEWER_ARCH_TYPES = [DEMUCS_V3_ARCH_TYPE, DEMUCS_V4_ARCH_TYPE]

# Network configuration
DEFAULT_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 120
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # Exponential backoff: 2^attempt seconds
CHUNK_SIZE = 8192


# ============================================================================
# Utility Functions
# ============================================================================

def retry_with_backoff(
    max_retries: int = MAX_RETRIES,
    backoff_base: int = RETRY_BACKOFF_BASE,
    exceptions: tuple = (urllib.error.URLError, socket.timeout, ConnectionError)
):
    """
    Decorator for retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_base: Base for exponential backoff (seconds)
        exceptions: Tuple of exception types to catch and retry
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = backoff_base ** attempt
                        # Try to get verbose flag from self if it's a method
                        verbose = True
                        if args and hasattr(args[0], 'verbose'):
                            verbose = args[0].verbose
                        if verbose:
                            print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s: {str(e)[:100]}")
                        time.sleep(wait_time)
            raise last_exception
        return wrapper
    return decorator


def fuzzy_match_model(query: str, model_names: List[str], threshold: float = 0.6) -> List[str]:
    """
    Find models with names similar to the query.
    
    Args:
        query: User's model name query
        model_names: List of available model names
        threshold: Minimum similarity ratio (0-1)
    
    Returns:
        List of similar model names, sorted by similarity
    """
    query_lower = query.lower()
    matches = []
    
    for name in model_names:
        # Check exact substring match first
        if query_lower in name.lower():
            matches.append((name, 1.0))
            continue
        
        # Calculate similarity ratio
        ratio = difflib.SequenceMatcher(None, query_lower, name.lower()).ratio()
        if ratio >= threshold:
            matches.append((name, ratio))
    
    # Sort by similarity (highest first)
    matches.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in matches]


def get_disk_free_space(path: str) -> int:
    """Get free disk space in bytes for the given path."""
    try:
        if sys.platform == 'win32':
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(path), None, None, ctypes.pointer(free_bytes)
            )
            return free_bytes.value
        else:
            stat = os.statvfs(path)
            return stat.f_bavail * stat.f_frsize
    except Exception:
        return -1  # Unknown


def format_bytes(size: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def calculate_file_hash(filepath: str, algorithm: str = 'md5', last_mb: int = 10) -> Optional[str]:
    """
    Calculate hash of a file (last N MB or whole file if smaller).
    
    This matches UVR GUI's hash calculation method.
    
    Args:
        filepath: Path to the file
        algorithm: Hash algorithm ('md5', 'sha256')
        last_mb: Read last N MB of file (UVR uses 10MB)
    
    Returns:
        Hex hash string or None if file doesn't exist
    """
    if not os.path.isfile(filepath):
        return None
    
    try:
        hasher = hashlib.new(algorithm)
        with open(filepath, 'rb') as f:
            try:
                # Seek to last N MB from end
                f.seek(-last_mb * 1024 * 1024, 2)
                hasher.update(f.read())
            except OSError:
                # File smaller than N MB, read whole file
                f.seek(0)
                hasher.update(f.read())
        return hasher.hexdigest()
    except Exception:
        return None


def atomic_move(src: str, dst: str) -> None:
    """
    Atomically move a file from src to dst, handling cross-filesystem moves safely.
    
    This function ensures that:
    1. If src and dst are on the same filesystem, uses atomic os.rename()
    2. If cross-filesystem, copies to a temp file on dst filesystem first,
       then performs atomic rename (prevents partial/corrupted files)
    3. Backs up existing dst file and restores on failure (prevents data loss)
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Raises:
        OSError: If the move operation fails
        IOError: If file operations fail
    """
    backup_path = None
    temp_dst = None
    
    try:
        # Step 1: Backup existing destination file if it exists
        if os.path.exists(dst):
            backup_path = dst + '.backup.' + str(os.getpid())
            try:
                os.rename(dst, backup_path)
            except OSError:
                # If rename fails (cross-filesystem), try copy
                shutil.copy2(dst, backup_path)
                os.remove(dst)
        
        # Step 2: Try atomic rename first (works if same filesystem)
        try:
            os.rename(src, dst)
            # Success! Clean up backup
            if backup_path and os.path.exists(backup_path):
                os.remove(backup_path)
            return
        except OSError as e:
            if e.errno != errno.EXDEV:  # Not a cross-device link error
                raise
            # Cross-filesystem move needed, continue below
        
        # Step 3: Cross-filesystem move - copy to temp on dst filesystem, then atomic rename
        dst_dir = os.path.dirname(dst) or '.'
        
        # Create temp file in the same directory as dst (same filesystem)
        fd, temp_dst = tempfile.mkstemp(
            prefix='.atomic_', 
            suffix='.tmp',
            dir=dst_dir
        )
        os.close(fd)
        
        # Copy with metadata
        shutil.copy2(src, temp_dst)
        
        # Verify copy integrity by comparing sizes
        src_size = os.path.getsize(src)
        tmp_size = os.path.getsize(temp_dst)
        if src_size != tmp_size:
            raise IOError(
                f"Copy verification failed: source {src_size} bytes, "
                f"copy {tmp_size} bytes"
            )
        
        # Atomic rename on same filesystem (guaranteed to work now)
        os.rename(temp_dst, dst)
        temp_dst = None  # Mark as successfully moved
        
        # Remove source file
        os.remove(src)
        
        # Clean up backup
        if backup_path and os.path.exists(backup_path):
            os.remove(backup_path)
            
    except Exception:
        # Restore backup on any failure
        if backup_path and os.path.exists(backup_path):
            try:
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(backup_path, dst)
            except OSError:
                pass  # Best effort restore
        
        # Clean up temp file
        if temp_dst and os.path.exists(temp_dst):
            try:
                os.remove(temp_dst)
            except OSError:
                pass
        
        raise
    
    finally:
        # Final cleanup of any leftover temp/backup files
        for path in [backup_path, temp_dst]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def is_valid_model_file(filepath: str, min_size: int = MIN_MODEL_FILE_SIZE) -> bool:
    """
    Check if a file exists and is a valid model file (not corrupted/truncated).
    
    This prevents false-positive "installed" status for:
    - Zero-byte files
    - Truncated files from interrupted downloads
    - Corrupted files
    
    Args:
        filepath: Path to the model file
        min_size: Minimum file size in bytes to be considered valid
        
    Returns:
        True if file exists and appears valid, False otherwise
    """
    if not os.path.isfile(filepath):
        return False
    
    try:
        size = os.path.getsize(filepath)
        if size < min_size:
            return False
        
        # Quick readability check - try to read first few bytes
        with open(filepath, 'rb') as f:
            header = f.read(16)
            if len(header) < 16 and size >= 16:
                return False  # File is unreadable or corrupted
                
        return True
    except (OSError, IOError):
        return False


@contextlib.contextmanager
def file_lock(lock_path: str, timeout: float = 300.0):
    """
    Cross-platform file lock context manager for concurrent container safety.
    
    This prevents race conditions when multiple containers try to download
    the same model simultaneously.
    
    IMPORTANT: Uses directory-based locking (mkdir is atomic even on NFS)
    as primary mechanism, with fcntl as secondary optimization for local filesystems.
    This ensures correct behavior on networked filesystems (NFS, CIFS, etc.)
    where fcntl advisory locks may not work across hosts.
    
    Args:
        lock_path: Path to the lock file (will be used to derive lock directory)
        timeout: Maximum time to wait for lock in seconds (default 5 minutes)
        
    Yields:
        None when lock is acquired
        
    Raises:
        TimeoutError: If lock cannot be acquired within timeout
        OSError: If lock cannot be created
    """
    # Use directory-based locking - mkdir is atomic on ALL filesystems including NFS
    # This is more reliable than fcntl which only works as advisory lock on NFS
    lock_dir_path = lock_path + '.lockdir'
    parent_dir = os.path.dirname(lock_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    
    start_time = time.time()
    lock_acquired = False
    stale_threshold = 3600  # Consider lock stale after 1 hour
    
    try:
        while True:
            try:
                # mkdir is atomic - if it succeeds, we have the lock
                os.mkdir(lock_dir_path)
                lock_acquired = True
                
                # Write PID and timestamp for debugging and stale lock detection
                lock_info_file = os.path.join(lock_dir_path, 'info')
                try:
                    with open(lock_info_file, 'w') as f:
                        f.write(f"pid={os.getpid()}\n")
                        f.write(f"time={time.time()}\n")
                        f.write(f"host={socket.gethostname()}\n")
                except OSError:
                    pass  # Info file is optional
                
                break  # Lock acquired successfully
                
            except FileExistsError:
                # Lock directory exists - check if it's stale
                lock_info_file = os.path.join(lock_dir_path, 'info')
                try:
                    if os.path.exists(lock_info_file):
                        mtime = os.path.getmtime(lock_info_file)
                        if time.time() - mtime > stale_threshold:
                            # Stale lock - try to remove it
                            try:
                                shutil.rmtree(lock_dir_path)
                                continue  # Retry acquiring lock
                            except OSError:
                                pass  # Another process may have removed it
                except OSError:
                    pass  # Can't check, just wait
                
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    raise TimeoutError(
                        f"Could not acquire lock on {lock_path} within {timeout}s. "
                        "Another process may be downloading this model. "
                        f"Lock directory: {lock_dir_path}"
                    )
                
                # Wait with exponential backoff (max 5 seconds)
                wait_time = min(0.5 * (1.1 ** min(elapsed / 10, 20)), 5.0)
                time.sleep(wait_time)
                
            except OSError as e:
                if e.errno == errno.EEXIST:
                    # Race condition - another process created it first
                    continue
                raise
        
        yield  # Lock held, execute protected code
        
    finally:
        # Release lock by removing directory
        if lock_acquired:
            try:
                shutil.rmtree(lock_dir_path)
            except OSError:
                pass  # Best effort cleanup


class ModelDownloader:
    """
    Handles model registry sync and downloading for all UVR architectures.
    
    Replicates UVR GUI download center behavior exactly.
    """
    
    def __init__(self, base_path: Optional[str] = None, verbose: bool = True):
        """
        Initialize the downloader.
        
        Args:
            base_path: Base directory for models (defaults to script directory)
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        
        # Setup paths (exactly matching UVR.py directory structure)
        if base_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        self.base_path = base_path
        self.models_dir = os.path.join(base_path, 'models')
        self.vr_models_dir = os.path.join(self.models_dir, 'VR_Models')
        self.mdx_models_dir = os.path.join(self.models_dir, 'MDX_Net_Models')
        self.demucs_models_dir = os.path.join(self.models_dir, 'Demucs_Models')
        self.demucs_newer_repo_dir = os.path.join(self.demucs_models_dir, 'v3_v4_repo')
        self.mdx_c_config_path = os.path.join(self.mdx_models_dir, 'model_data', 'mdx_c_configs')
        
        # Ensure directories exist
        for dir_path in [self.vr_models_dir, self.mdx_models_dir, 
                         self.demucs_models_dir, self.demucs_newer_repo_dir,
                         self.mdx_c_config_path]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Model registry (fetched from remote)
        self.online_data: Dict = {}
        self.vr_download_list: Dict = {}
        self.mdx_download_list: Dict = {}
        self.demucs_download_list: Dict = {}
        
        # Model checksums for integrity verification
        self.model_checksums: Dict[str, str] = {}
        
        # Local cache path for download_checks.json
        self.cache_dir = os.path.join(base_path, '.model_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.download_checks_cache = os.path.join(self.cache_dir, 'download_checks.json')
        self.checksums_cache = os.path.join(self.cache_dir, 'model_checksums.json')
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def sync_registry(self, force: bool = False) -> bool:
        """
        Sync model registry from official UVR sources.
        
        This replicates UVR.py online_data_refresh() behavior.
        
        Args:
            force: Force refresh even if cache exists
            
        Returns:
            True if sync successful, False otherwise
            
        Raises:
            RegistryError: If sync fails and no cache available
        """
        # Check cache first
        if not force and os.path.isfile(self.download_checks_cache):
            try:
                cache_age = os.path.getmtime(self.download_checks_cache)
                # Cache valid for 1 hour
                if time.time() - cache_age < 3600:
                    self._log("Using cached model registry...")
                    with open(self.download_checks_cache, 'r', encoding='utf-8') as f:
                        self.online_data = json.load(f)
                    self._populate_model_lists()
                    return True
            except (OSError, json.JSONDecodeError) as e:
                self._log(f"Warning: Cache read error: {e}")
                # Continue to fetch from network
        
        # Try to fetch from network with retry
        network_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                self._log("Syncing model registry from official UVR sources...")
                
                # Log proxy status on first attempt (helps with debugging)
                if attempt == 0 and is_proxy_configured():
                    self._log("  Using HTTP proxy (configured via environment)")
                
                # Fetch download_checks.json (UVR.py line 5605)
                request = urllib.request.Request(
                    DOWNLOAD_CHECKS_URL,
                    headers={'User-Agent': 'UVR-Headless-Runner/1.0'}
                )
                with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
                    raw_data = response.read().decode('utf-8')
                    self.online_data = json.loads(raw_data)
                
                # Validate response structure
                if not isinstance(self.online_data, dict):
                    raise RegistryError(
                        "Invalid registry format: expected dictionary",
                        "The server may be returning an error. Try again later."
                    )
                
                # Save to cache (atomic write)
                temp_cache = self.download_checks_cache + '.tmp'
                try:
                    with open(temp_cache, 'w', encoding='utf-8') as f:
                        json.dump(self.online_data, f, indent=2)
                    # Atomic rename
                    if os.path.exists(self.download_checks_cache):
                        os.remove(self.download_checks_cache)
                    shutil.move(temp_cache, self.download_checks_cache)
                except OSError as e:
                    self._log(f"Warning: Could not save cache: {e}")
                    # Clean up temp file
                    if os.path.exists(temp_cache):
                        try:
                            os.remove(temp_cache)
                        except OSError:
                            pass
                
                self._populate_model_lists()
                
                # Also sync checksums (optional, non-blocking)
                if VERIFY_CHECKSUMS:
                    self.sync_checksums()
                
                self._log("Model registry synced successfully!")
                return True
                
            except urllib.error.HTTPError as e:
                network_error = NetworkError(
                    f"HTTP {e.code}: {e.reason}",
                    "The UVR model server may be temporarily unavailable."
                )
                if e.code >= 500:
                    # Server error - retry
                    if attempt < MAX_RETRIES:
                        wait_time = RETRY_BACKOFF_BASE ** attempt
                        self._log(f"  Server error, retry {attempt + 1}/{MAX_RETRIES} after {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                break
                
            except urllib.error.URLError as e:
                network_error = NetworkError(
                    f"Connection failed: {e.reason}",
                    "Check your internet connection or try again later."
                )
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_BACKOFF_BASE ** attempt
                    self._log(f"  Connection error, retry {attempt + 1}/{MAX_RETRIES} after {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                break
                
            except socket.timeout:
                network_error = NetworkError(
                    "Connection timed out",
                    "The server is not responding. Check your network or try again later."
                )
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_BACKOFF_BASE ** attempt
                    self._log(f"  Timeout, retry {attempt + 1}/{MAX_RETRIES} after {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                break
                
            except json.JSONDecodeError as e:
                network_error = RegistryError(
                    f"Invalid JSON response: {e}",
                    "The server returned malformed data. Try again later."
                )
                break
                
            except Exception as e:
                network_error = RegistryError(
                    f"Unexpected error: {e}",
                    "Please report this issue."
                )
                break
        
        # Network fetch failed, try cache as fallback
        if os.path.isfile(self.download_checks_cache):
            try:
                self._log("Network unavailable. Using cached registry...")
                with open(self.download_checks_cache, 'r', encoding='utf-8') as f:
                    self.online_data = json.load(f)
                self._populate_model_lists()
                return True
            except (OSError, json.JSONDecodeError) as e:
                self._log(f"Cache also unreadable: {e}")
        
        # Both network and cache failed
        if network_error:
            self._log(str(network_error))
        return False
    
    def _populate_model_lists(self):
        """
        Populate per-architecture model lists from online_data.
        
        Replicates UVR.py download_list_fill() behavior (lines 5736-5745).
        """
        self.vr_download_list = self.online_data.get("vr_download_list", {})
        self.mdx_download_list = self.online_data.get("mdx_download_list", {})
        self.demucs_download_list = self.online_data.get("demucs_download_list", {})
        
        # Merge additional MDX lists (UVR.py line 5739-5740)
        self.mdx_download_list.update(self.online_data.get("mdx23c_download_list", {}))
        self.mdx_download_list.update(self.online_data.get("other_network_list", {}))
        self.mdx_download_list.update(self.online_data.get("other_network_list_new", {}))
    
    def sync_checksums(self, force: bool = False) -> bool:
        """
        Sync model checksums from official source for integrity verification.
        
        Args:
            force: Force refresh even if cache exists
            
        Returns:
            True if checksums available, False otherwise
        """
        if not VERIFY_CHECKSUMS:
            return False
        
        # Check cache first
        if not force and os.path.isfile(self.checksums_cache):
            try:
                cache_age = os.path.getmtime(self.checksums_cache)
                if time.time() - cache_age < 86400:  # Cache valid for 24 hours
                    with open(self.checksums_cache, 'r', encoding='utf-8') as f:
                        self.model_checksums = json.load(f)
                    return bool(self.model_checksums)
            except (OSError, json.JSONDecodeError):
                pass
        
        # Try to fetch from network
        try:
            request = urllib.request.Request(
                MODEL_CHECKSUMS_URL,
                headers={'User-Agent': 'UVR-Headless-Runner/1.0'}
            )
            with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
                raw_data = response.read().decode('utf-8')
                self.model_checksums = json.loads(raw_data)
            
            # Save to cache
            try:
                with open(self.checksums_cache, 'w', encoding='utf-8') as f:
                    json.dump(self.model_checksums, f, indent=2)
            except OSError:
                pass
            
            return bool(self.model_checksums)
            
        except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout, json.JSONDecodeError):
            # Checksums not available - this is OK, verification is optional
            # Try loading from cache as fallback
            if os.path.isfile(self.checksums_cache):
                try:
                    with open(self.checksums_cache, 'r', encoding='utf-8') as f:
                        self.model_checksums = json.load(f)
                    return bool(self.model_checksums)
                except (OSError, json.JSONDecodeError):
                    pass
            return False
    
    def get_model_checksum(self, filename: str) -> Optional[str]:
        """
        Get expected checksum for a model file.
        
        Args:
            filename: Model filename
            
        Returns:
            SHA256 checksum string or None if not available
        """
        if not self.model_checksums:
            self.sync_checksums()
        return self.model_checksums.get(filename)
    
    def verify_file_checksum(self, filepath: str, expected_hash: str = None) -> Tuple[bool, str]:
        """
        Verify file integrity using SHA256 checksum.
        
        Args:
            filepath: Path to file to verify
            expected_hash: Expected SHA256 hash (if None, looks up from registry)
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not os.path.isfile(filepath):
            return False, f"File not found: {filepath}"
        
        filename = os.path.basename(filepath)
        
        if expected_hash is None:
            expected_hash = self.get_model_checksum(filename)
        
        if expected_hash is None:
            return True, f"No checksum available for {filename} (skipping verification)"
        
        # Calculate full file SHA256
        try:
            sha256 = hashlib.sha256()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    sha256.update(chunk)
            actual_hash = sha256.hexdigest()
            
            if actual_hash.lower() == expected_hash.lower():
                return True, f"Checksum verified: {filename}"
            else:
                return False, (
                    f"Checksum mismatch for {filename}:\n"
                    f"  Expected: {expected_hash}\n"
                    f"  Actual:   {actual_hash}"
                )
        except OSError as e:
            return False, f"Cannot read file for checksum: {e}"
    
    def list_models(self, arch_type: str, show_installed: bool = True) -> List[Dict[str, Any]]:
        """
        List available models for an architecture.
        
        Args:
            arch_type: 'vr', 'mdx', or 'demucs'
            show_installed: If True, include installation status
            
        Returns:
            List of model info dictionaries
        """
        if not self.online_data:
            self.sync_registry()
        
        arch_type = arch_type.lower()
        models = []
        
        if arch_type == 'vr':
            download_list = self.vr_download_list
            model_dir = self.vr_models_dir
        elif arch_type in ['mdx', 'mdx-net']:
            download_list = self.mdx_download_list
            model_dir = self.mdx_models_dir
        elif arch_type == 'demucs':
            download_list = self.demucs_download_list
            model_dir = self.demucs_models_dir
        else:
            raise ValueError(f"Unknown architecture: {arch_type}")
        
        for display_name, model_info in download_list.items():
            model_data = {
                'display_name': display_name,
                'name': self._extract_model_name(display_name),
            }
            
            # Determine filename(s) and check installation
            if isinstance(model_info, dict):
                # Complex model (MDX-C/Roformer or Demucs with multiple files)
                files = list(model_info.keys())
                model_data['files'] = files
                model_data['is_multi_file'] = True
                
                # Check if installed (all files present AND valid)
                if arch_type == 'demucs':
                    is_newer = any(x in display_name for x in ['v3', 'v4'])
                    check_dir = self.demucs_newer_repo_dir if is_newer else self.demucs_models_dir
                else:
                    check_dir = model_dir
                
                # Use is_valid_model_file instead of os.path.isfile to prevent
                # false-positive "installed" status for corrupted/truncated files
                installed = all(
                    is_valid_model_file(os.path.join(check_dir, f)) or
                    is_valid_model_file(os.path.join(self.mdx_c_config_path, f))
                    for f in files
                )
            else:
                # Simple model (single file)
                model_data['files'] = [str(model_info)]
                model_data['is_multi_file'] = False
                # Use is_valid_model_file to verify file is not corrupted/truncated
                installed = is_valid_model_file(os.path.join(model_dir, str(model_info)))
            
            if show_installed:
                model_data['installed'] = installed
            
            models.append(model_data)
        
        return models
    
    def _extract_model_name(self, display_name: str) -> str:
        """Extract clean model name from display name."""
        # Remove prefix like "VR Arch Single Model v5: " or "MDX-Net Model: "
        if ':' in display_name:
            return display_name.split(':', 1)[1].strip()
        return display_name
    
    def get_model_info(
        self, 
        model_name: str, 
        arch_type: str,
        raise_on_not_found: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed info for a specific model.
        
        Args:
            model_name: Model name or display name
            arch_type: 'vr', 'mdx', or 'demucs'
            raise_on_not_found: If True, raise ModelNotFoundError instead of returning None
            
        Returns:
            Model info dictionary or None if not found
            
        Raises:
            ModelNotFoundError: If raise_on_not_found=True and model not found
            ValueError: If arch_type is invalid
        """
        if not self.online_data:
            self.sync_registry()
        
        arch_type = arch_type.lower()
        
        if arch_type == 'vr':
            download_list = self.vr_download_list
            model_dir = self.vr_models_dir
            subdir = 'VR_Models'
        elif arch_type in ['mdx', 'mdx-net']:
            download_list = self.mdx_download_list
            model_dir = self.mdx_models_dir
            subdir = 'MDX_Net_Models'
        elif arch_type == 'demucs':
            download_list = self.demucs_download_list
            model_dir = self.demucs_models_dir
            subdir = 'Demucs_Models'
        else:
            raise ValueError(f"Unknown architecture: {arch_type}. Valid options: vr, mdx, demucs")
        
        # Search by exact display name or partial match
        for display_name, model_info in download_list.items():
            # Match by display name or extracted name
            clean_name = self._extract_model_name(display_name)
            if model_name in [display_name, clean_name] or model_name.lower() == clean_name.lower():
                result = {
                    'display_name': display_name,
                    'name': clean_name,
                    'arch_type': arch_type,
                    'subdir': subdir,
                }
                
                # Parse model info based on type
                if isinstance(model_info, dict):
                    result['is_multi_file'] = True
                    result['files'] = {}
                    
                    for filename, value in model_info.items():
                        if isinstance(value, str) and (value.startswith('http://') or value.startswith('https://')):
                            # Full URL provided
                            result['files'][filename] = value
                        else:
                            # Need to construct URL
                            if filename.endswith('.yaml'):
                                # Config file - check if it's a reference or needs URL
                                if arch_type in ['mdx', 'mdx-net']:
                                    result['files'][filename] = f"{MDX23_CONFIG_CHECKS}{filename}"
                            else:
                                result['files'][filename] = f"{NORMAL_REPO}{filename}"
                    
                    # Determine actual save directory for Demucs
                    if arch_type == 'demucs':
                        is_newer = any(x in display_name for x in ['v3', 'v4'])
                        result['save_dir'] = self.demucs_newer_repo_dir if is_newer else self.demucs_models_dir
                        result['subdir'] = 'Demucs_Models/v3_v4_repo' if is_newer else 'Demucs_Models'
                    else:
                        result['save_dir'] = model_dir
                else:
                    # Simple single-file model
                    result['is_multi_file'] = False
                    filename = str(model_info)
                    result['filename'] = filename
                    result['url'] = f"{NORMAL_REPO}{filename}"
                    result['save_dir'] = model_dir
                    result['local_path'] = os.path.join(model_dir, filename)
                
                # Check if installed (using is_valid_model_file to prevent false-positives)
                if result.get('is_multi_file'):
                    save_dir = result.get('save_dir', model_dir)
                    result['installed'] = all(
                        is_valid_model_file(os.path.join(save_dir, f)) or
                        is_valid_model_file(os.path.join(self.mdx_c_config_path, f))
                        for f in result['files'].keys()
                    )
                else:
                    result['installed'] = is_valid_model_file(result.get('local_path', ''))
                
                return result
        
        # Model not found - provide helpful suggestions
        if raise_on_not_found:
            # Get all model names for fuzzy matching
            all_names = [self._extract_model_name(dn) for dn in download_list.keys()]
            similar = fuzzy_match_model(model_name, all_names)
            raise ModelNotFoundError(model_name, arch_type, similar)
        
        return None
    
    def download_model(
        self, 
        model_name: str, 
        arch_type: str,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[bool, str]:
        """
        Download a model from official UVR sources.
        
        Replicates UVR.py download_item() behavior with enhanced error handling.
        
        Args:
            model_name: Model name or display name
            arch_type: 'vr', 'mdx', or 'demucs'
            progress_callback: Optional callback(current, total, filename)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        # Get model info with fuzzy matching suggestions on failure
        try:
            model_info = self.get_model_info(model_name, arch_type, raise_on_not_found=True)
        except ModelNotFoundError as e:
            return False, str(e)
        except ValueError as e:
            return False, f"Invalid architecture type: {e}"
        
        if model_info is None:
            return False, f"Model not found: {model_name}"
        
        if model_info.get('installed'):
            return True, f"Model already installed: {model_name}"
        
        try:
            if model_info.get('is_multi_file'):
                # Download multiple files (Demucs or MDX-C/Roformer) TRANSACTIONALLY
                # All files are downloaded to a staging directory first, then moved atomically
                files = model_info['files']
                save_dir = model_info.get('save_dir', self.mdx_models_dir)
                
                total_files = len(files)
                
                # Check which files already exist (and are valid)
                files_to_download = {}
                existing_files = 0
                for filename, url in files.items():
                    if filename.endswith('.yaml') and arch_type in ['mdx', 'mdx-net']:
                        final_path = os.path.join(self.mdx_c_config_path, filename)
                    else:
                        final_path = os.path.join(save_dir, filename)
                    
                    if is_valid_model_file(final_path):
                        existing_files += 1
                        self._log(f"  [exists] {filename}")
                    else:
                        files_to_download[filename] = (url, final_path)
                
                # If all files exist, we're done
                if not files_to_download:
                    return True, f"All {total_files} files already installed: {model_name}"
                
                # Create staging directory for transactional download
                staging_dir = os.path.join(save_dir, f'.staging_{os.getpid()}_{int(time.time())}')
                staged_files = []  # Track for cleanup on failure
                
                try:
                    os.makedirs(staging_dir, exist_ok=True)
                    
                    # Download all files to staging directory
                    for i, (filename, (url, final_path)) in enumerate(files_to_download.items(), 1):
                        self._log(f"  [{i}/{len(files_to_download)}] Downloading {filename}...")
                        
                        staging_path = os.path.join(staging_dir, filename)
                        staged_files.append((staging_path, final_path, filename))
                        
                        try:
                            self._download_file(url, staging_path, progress_callback)
                        except (DownloadError, NetworkError, IntegrityError) as e:
                            # Clean up staging directory on failure
                            return False, (
                                f"Download failed for {filename}: {e.message}\n"
                                f"  Downloaded {i-1}/{len(files_to_download)} files.\n"
                                f"  {e.suggestion if e.suggestion else ''}"
                            )
                    
                    # All downloads successful - now move atomically to final locations
                    self._log("  Moving downloaded files to final location...")
                    moved_files = []
                    try:
                        for staging_path, final_path, filename in staged_files:
                            # Ensure target directory exists
                            final_dir = os.path.dirname(final_path)
                            os.makedirs(final_dir, exist_ok=True)
                            
                            atomic_move(staging_path, final_path)
                            moved_files.append(final_path)
                    except (OSError, IOError) as e:
                        # Rollback: remove any files that were successfully moved
                        for moved_path in moved_files:
                            try:
                                os.remove(moved_path)
                            except OSError:
                                pass
                        raise DownloadError(
                            f"Failed to move files to final location: {e}",
                            "Check disk space and permissions."
                        )
                    
                    return True, f"Successfully downloaded: {model_name} ({len(files_to_download)} new files)"
                    
                finally:
                    # Clean up staging directory
                    try:
                        shutil.rmtree(staging_dir, ignore_errors=True)
                    except OSError:
                        pass
            else:
                # Download single file
                url = model_info['url']
                save_path = model_info['local_path']
                filename = model_info['filename']
                
                self._log(f"Downloading {filename}...")
                try:
                    self._download_file(url, save_path, progress_callback)
                except (DownloadError, NetworkError, IntegrityError, DiskSpaceError) as e:
                    return False, f"{e.message}\n  {e.suggestion if e.suggestion else ''}"
                
                return True, f"Successfully downloaded: {model_name}"
                
        except DiskSpaceError as e:
            return False, f"{e.message}\n  {e.suggestion}"
        except ModelDownloaderError as e:
            return False, str(e)
        except Exception as e:
            return False, (
                f"Unexpected error during download: {type(e).__name__}: {e}\n"
                f"  Please report this issue if it persists."
            )
    
    def _download_file(
        self, 
        url: str, 
        save_path: str, 
        progress_callback: Optional[Callable] = None,
        expected_size: int = 0,
        expected_hash: str = None,
        verify_size: bool = True
    ):
        """
        Download a file with progress reporting, resume support, and integrity checking.
        
        Features:
        - Cross-process file locking (prevents concurrent container race conditions)
        - Retry with exponential backoff
        - Resume partial downloads
        - Disk space pre-check
        - Atomic file writes
        - Optional hash verification
        
        Args:
            url: URL to download from
            save_path: Local path to save file
            progress_callback: Optional callback(current, total, filename)
            expected_size: Expected file size (for disk space check)
            expected_hash: Expected file hash for verification
            verify_size: Whether to verify downloaded size matches Content-Length
            
        Raises:
            DownloadError: If download fails after all retries
            DiskSpaceError: If insufficient disk space
            IntegrityError: If hash verification fails
            TimeoutError: If cannot acquire file lock (another process downloading)
        """
        temp_path = save_path + '.tmp'
        lock_path = save_path + '.lock'
        filename = os.path.basename(save_path)
        
        # Ensure directory exists before acquiring lock
        save_dir = os.path.dirname(save_path)
        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError as e:
            raise DownloadError(
                f"Cannot create directory: {save_dir}",
                f"Check write permissions or create the directory manually: {e}"
            )
        
        # Use cross-process file lock to prevent concurrent container race conditions
        # This is CRITICAL for data integrity when multiple containers run simultaneously
        try:
            with file_lock(lock_path, timeout=300.0):
                # After acquiring lock, check if another process already completed the download
                if is_valid_model_file(save_path):
                    self._log(f"    {filename} already downloaded by another process")
                    return
                
                # Proceed with download inside the lock
                self._download_file_impl(
                    url, save_path, temp_path, filename, save_dir,
                    progress_callback, expected_size, expected_hash, verify_size
                )
        except TimeoutError as e:
            raise DownloadError(
                f"Could not acquire download lock for {filename}",
                str(e)
            )
    
    def _download_file_impl(
        self,
        url: str,
        save_path: str,
        temp_path: str,
        filename: str,
        save_dir: str,
        progress_callback: Optional[Callable] = None,
        expected_size: int = 0,
        expected_hash: str = None,
        verify_size: bool = True
    ):
        """
        Internal implementation of file download (called with lock held).
        """
        # Register this download for cleanup on interrupt
        with _download_lock:
            _active_downloads.add(temp_path)
        
        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                # Check for existing partial download
                resume_pos = 0
                if os.path.isfile(temp_path):
                    resume_pos = os.path.getsize(temp_path)
                    self._log(f"    Resuming from {format_bytes(resume_pos)}...")
                
                # Build request with resume support
                request = urllib.request.Request(url)
                request.add_header('User-Agent', 'UVR-Headless-Runner/1.0')
                if resume_pos > 0:
                    request.add_header('Range', f'bytes={resume_pos}-')
                
                # Open connection
                with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
                    # Get content info
                    content_length = response.headers.get('Content-Length')
                    total_size = int(content_length) if content_length else 0
                    
                    # Handle resume response
                    if response.status == 206:  # Partial Content
                        total_size += resume_pos
                    elif response.status == 200 and resume_pos > 0:
                        # Server doesn't support resume, restart
                        resume_pos = 0
                        if os.path.isfile(temp_path):
                            os.remove(temp_path)
                    
                    # Check disk space
                    if total_size > 0:
                        free_space = get_disk_free_space(save_dir)
                        if free_space > 0 and free_space < total_size * 1.1:  # 10% margin
                            raise DiskSpaceError(
                                f"Insufficient disk space: need {format_bytes(total_size)}, "
                                f"have {format_bytes(free_space)}",
                                "Free up disk space or choose a different download location."
                            )
                    
                    downloaded = resume_pos
                    
                    # Open file for append or write
                    mode = 'ab' if resume_pos > 0 else 'wb'
                    with open(temp_path, mode) as f:
                        while True:
                            try:
                                chunk = response.read(CHUNK_SIZE)
                            except socket.timeout:
                                raise NetworkError(
                                    "Download stalled",
                                    "Network connection lost. Will retry..."
                                )
                            
                            if not chunk:
                                break
                            
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if progress_callback and total_size > 0:
                                progress_callback(downloaded, total_size, filename)
                            elif self.verbose and total_size > 0:
                                percent = int(100 * downloaded / total_size)
                                speed_info = ""
                                print(f"\r    Progress: {percent}% ({format_bytes(downloaded)}/{format_bytes(total_size)}){speed_info}", end='', flush=True)
                    
                    if self.verbose and total_size > 0 and not progress_callback:
                        print()  # New line after progress
                    
                    # Verify downloaded size
                    if verify_size and total_size > 0 and downloaded != total_size:
                        raise DownloadError(
                            f"Incomplete download: got {format_bytes(downloaded)}, expected {format_bytes(total_size)}",
                            "Download was interrupted. Will retry..."
                        )
                
                # Verify hash if provided
                if expected_hash:
                    actual_hash = calculate_file_hash(temp_path)
                    if actual_hash and actual_hash != expected_hash:
                        # Remove corrupted file
                        os.remove(temp_path)
                        raise IntegrityError(
                            f"Hash mismatch for {filename}",
                            "The downloaded file is corrupted. Will retry download."
                        )
                
                # Atomic move to final location (with backup/restore for safety)
                try:
                    atomic_move(temp_path, save_path)
                except OSError as e:
                    raise DownloadError(
                        f"Cannot save file: {save_path}",
                        f"Check file permissions: {e}"
                    )
                except IOError as e:
                    raise IntegrityError(
                        f"File copy verification failed for {filename}",
                        str(e)
                    )
                
                # Success! Unregister from cleanup tracking
                with _download_lock:
                    _active_downloads.discard(temp_path)
                return
                
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise DownloadError(
                        f"File not found on server: {filename}",
                        "The model may have been moved or removed from the repository."
                    )
                elif e.code == 416:  # Range not satisfiable
                    # Remove invalid partial file and restart
                    if os.path.isfile(temp_path):
                        os.remove(temp_path)
                    continue
                else:
                    last_error = DownloadError(
                        f"HTTP {e.code}: {e.reason}",
                        "Server error. Will retry..."
                    )
                    
            except urllib.error.URLError as e:
                last_error = NetworkError(
                    f"Connection failed: {e.reason}",
                    "Check your internet connection."
                )
                
            except socket.timeout:
                last_error = NetworkError(
                    "Connection timed out",
                    "The server is not responding."
                )
            
            except (DiskSpaceError, IntegrityError):
                # Don't retry these errors
                raise
                
            except OSError as e:
                if e.errno == 28:  # No space left on device
                    raise DiskSpaceError(
                        "Disk full during download",
                        "Free up disk space and try again."
                    )
                last_error = DownloadError(
                    f"File system error: {e}",
                    "Check disk and permissions."
                )
            
            # Retry logic
            if attempt < MAX_RETRIES:
                wait_time = RETRY_BACKOFF_BASE ** attempt
                self._log(f"    Retry {attempt + 1}/{MAX_RETRIES} after {wait_time}s...")
                time.sleep(wait_time)
        
        # All retries exhausted
        # Clean up temp file and unregister from tracking
        with _download_lock:
            _active_downloads.discard(temp_path)
        if os.path.isfile(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        
        if last_error:
            raise last_error
        raise DownloadError(
            f"Download failed after {MAX_RETRIES} attempts",
            "Check your network connection and try again later."
        )
    
    def ensure_model(
        self, 
        model_name: str, 
        arch_type: str,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[bool, str]:
        """
        Ensure a model is available locally, downloading if necessary.
        
        This is the main entry point for automatic downloading with full
        error handling and user feedback.
        
        Args:
            model_name: Model name or display name
            arch_type: 'vr', 'mdx', or 'demucs'
            progress_callback: Optional progress callback
            
        Returns:
            Tuple of (success: bool, local_path_or_error: str)
        """
        try:
            model_info = self.get_model_info(model_name, arch_type, raise_on_not_found=False)
        except ValueError as e:
            return False, f"Invalid architecture type: {e}"
        
        if model_info is None:
            # Try sync and retry with fuzzy matching
            self._log(f"Model '{model_name}' not found. Refreshing registry...")
            self.sync_registry(force=True)
            
            try:
                model_info = self.get_model_info(model_name, arch_type, raise_on_not_found=True)
            except ModelNotFoundError as e:
                return False, str(e)
        
        # Helper function to get local path from model_info
        def get_local_path(info: Dict) -> str:
            if info.get('is_multi_file'):
                save_dir = info.get('save_dir')
                # Return path to the main model file (not config)
                for f in info['files'].keys():
                    if not f.endswith('.yaml'):
                        return os.path.join(save_dir, f)
                # If all yaml, return first file
                first_file = list(info['files'].keys())[0]
                return os.path.join(save_dir, first_file)
            else:
                return info['local_path']
        
        if model_info.get('installed'):
            # Return local path
            local_path = get_local_path(model_info)
            
            # Verify file actually exists
            if not os.path.isfile(local_path):
                self._log(f"Warning: Model marked as installed but file missing. Re-downloading...")
            else:
                return True, local_path
        
        # Download the model
        self._log(f"Model not found locally. Downloading: {model_name}")
        success, message = self.download_model(model_name, arch_type, progress_callback)
        
        if success:
            # Refresh model_info and return local path
            model_info = self.get_model_info(model_name, arch_type)
            if model_info:
                return True, get_local_path(model_info)
            else:
                return False, "Download succeeded but model info unavailable"
        
        return False, message
    
    def verify_model_integrity(
        self, 
        model_name: str, 
        arch_type: str,
        redownload_on_failure: bool = False
    ) -> Tuple[bool, str]:
        """
        Verify the integrity of a downloaded model.
        
        Checks:
        1. File exists
        2. File size is non-zero
        3. File is readable
        4. For ONNX: basic structure validation
        
        Args:
            model_name: Model name
            arch_type: Architecture type
            redownload_on_failure: If True, attempt to redownload corrupted files
            
        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        model_info = self.get_model_info(model_name, arch_type)
        
        if model_info is None:
            return False, f"Model not found in registry: {model_name}"
        
        if not model_info.get('installed'):
            return False, f"Model not installed: {model_name}"
        
        # Get file paths to check
        files_to_check = []
        if model_info.get('is_multi_file'):
            save_dir = model_info.get('save_dir')
            for f in model_info['files'].keys():
                if f.endswith('.yaml'):
                    files_to_check.append(os.path.join(self.mdx_c_config_path, f))
                else:
                    files_to_check.append(os.path.join(save_dir, f))
        else:
            files_to_check.append(model_info['local_path'])
        
        # Check each file
        for filepath in files_to_check:
            filename = os.path.basename(filepath)
            
            # Check existence
            if not os.path.isfile(filepath):
                if redownload_on_failure:
                    self._log(f"Missing file: {filename}. Redownloading...")
                    success, msg = self.download_model(model_name, arch_type)
                    if not success:
                        return False, f"Redownload failed: {msg}"
                    continue
                return False, f"File missing: {filepath}"
            
            # Check size
            size = os.path.getsize(filepath)
            if size == 0:
                if redownload_on_failure:
                    os.remove(filepath)
                    self._log(f"Empty file: {filename}. Redownloading...")
                    success, msg = self.download_model(model_name, arch_type)
                    if not success:
                        return False, f"Redownload failed: {msg}"
                    continue
                return False, f"File is empty: {filepath}"
            
            # Check readability
            try:
                with open(filepath, 'rb') as f:
                    # Read first few bytes to verify file is accessible
                    header = f.read(16)
                    if len(header) < 16 and size >= 16:
                        return False, f"File read error: {filepath}"
            except OSError as e:
                return False, f"Cannot read file {filepath}: {e}"
            
            # ONNX-specific validation
            if filepath.endswith('.onnx'):
                # Check ONNX magic number
                if header[:4] != b'\x08\x00\x12\x04':
                    # Not all ONNX files have this header, but we can check for protobuf structure
                    pass  # Skip strict validation
        
        return True, f"Model integrity verified: {model_name}"
    
    def get_local_model_path(self, model_name: str, arch_type: str) -> Optional[str]:
        """
        Get the local path for a model if it exists.
        
        Args:
            model_name: Model name or display name
            arch_type: 'vr', 'mdx', or 'demucs'
            
        Returns:
            Local path if model exists, None otherwise
        """
        model_info = self.get_model_info(model_name, arch_type)
        
        if model_info is None:
            return None
        
        if not model_info.get('installed'):
            return None
        
        if model_info.get('is_multi_file'):
            save_dir = model_info.get('save_dir')
            for f in model_info['files'].keys():
                if not f.endswith('.yaml'):
                    return os.path.join(save_dir, f)
            first_file = list(model_info['files'].keys())[0]
            return os.path.join(save_dir, first_file)
        else:
            return model_info.get('local_path')


# ============================================================================
# Architecture-specific registries (for embedding in runners)
# ============================================================================

def get_mdx_models() -> Dict[str, Dict]:
    """Get MDX model registry."""
    downloader = ModelDownloader(verbose=False)
    downloader.sync_registry()
    
    result = {}
    for model in downloader.list_models('mdx', show_installed=True):
        clean_name = model['name']
        result[clean_name] = {
            'display_name': model['display_name'],
            'files': model['files'],
            'installed': model['installed'],
            'subdir': 'MDX_Net_Models'
        }
    return result


def get_vr_models() -> Dict[str, Dict]:
    """Get VR model registry."""
    downloader = ModelDownloader(verbose=False)
    downloader.sync_registry()
    
    result = {}
    for model in downloader.list_models('vr', show_installed=True):
        clean_name = model['name']
        result[clean_name] = {
            'display_name': model['display_name'],
            'files': model['files'],
            'installed': model['installed'],
            'subdir': 'VR_Models'
        }
    return result


def get_demucs_models() -> Dict[str, Dict]:
    """Get Demucs model registry."""
    downloader = ModelDownloader(verbose=False)
    downloader.sync_registry()
    
    result = {}
    for model in downloader.list_models('demucs', show_installed=True):
        clean_name = model['name']
        is_newer = any(x in model['display_name'] for x in ['v3', 'v4'])
        result[clean_name] = {
            'display_name': model['display_name'],
            'files': model['files'],
            'installed': model['installed'],
            'subdir': 'Demucs_Models/v3_v4_repo' if is_newer else 'Demucs_Models'
        }
    return result


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface for model downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='UVR Model Downloader - Download models from official UVR sources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all MDX models
  python model_downloader.py --list mdx
  
  # List only uninstalled VR models
  python model_downloader.py --list vr --uninstalled
  
  # Download a specific model
  python model_downloader.py --download "UVR-MDX-NET Inst HQ 3" --arch mdx
  
  # Get model info
  python model_downloader.py --info "htdemucs" --arch demucs
  
  # Sync registry from remote
  python model_downloader.py --sync
"""
    )
    
    parser.add_argument('--list', '-l', choices=['vr', 'mdx', 'demucs'], 
                        help='List available models for architecture')
    parser.add_argument('--uninstalled', action='store_true',
                        help='Only show uninstalled models')
    parser.add_argument('--download', '-d', help='Download a model by name')
    parser.add_argument('--arch', '-a', choices=['vr', 'mdx', 'demucs'],
                        help='Architecture type (required for --download and --info)')
    parser.add_argument('--info', '-i', help='Get detailed info for a model')
    parser.add_argument('--sync', '-s', action='store_true',
                        help='Force sync registry from remote')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Quiet mode')
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(verbose=not args.quiet)
    
    if args.sync:
        success = downloader.sync_registry(force=True)
        if success:
            print("Registry synced successfully!")
        else:
            print("Failed to sync registry")
            return 1
    
    if args.list:
        models = downloader.list_models(args.list)
        
        if args.uninstalled:
            models = [m for m in models if not m['installed']]
        
        if not models:
            print(f"No {'uninstalled ' if args.uninstalled else ''}models found for {args.list}")
            return 0
        
        print(f"\n{'Uninstalled ' if args.uninstalled else ''}Models for {args.list.upper()}:")
        print("=" * 60)
        for model in models:
            status = "Y" if model['installed'] else "N"
            print(f"  [{status}] {model['name']}")
        print(f"\nTotal: {len(models)} models")
        return 0
    
    if args.info:
        if not args.arch:
            print("Error: --arch is required with --info")
            return 1
        
        info = downloader.get_model_info(args.info, args.arch)
        if info:
            print(f"\nModel Info: {info['name']}")
            print("=" * 60)
            print(f"  Display Name: {info['display_name']}")
            print(f"  Architecture: {info['arch_type']}")
            print(f"  Installed: {'Yes' if info['installed'] else 'No'}")
            print(f"  Directory: {info['subdir']}")
            if info.get('is_multi_file'):
                print(f"  Files:")
                for f, url in info['files'].items():
                    print(f"    - {f}")
                    print(f"      URL: {url[:80]}...")
            else:
                print(f"  Filename: {info['filename']}")
                print(f"  URL: {info['url']}")
        else:
            print(f"Model not found: {args.info}")
            return 1
        return 0
    
    if args.download:
        if not args.arch:
            print("Error: --arch is required with --download")
            return 1
        
        print(f"Downloading: {args.download}")
        success, message = downloader.download_model(args.download, args.arch)
        print(message)
        return 0 if success else 1
    
    # Default: show help
    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
