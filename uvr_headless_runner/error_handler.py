#!/usr/bin/env python3
"""
Error Handler Module for UVR Headless Runners
==============================================

Provides centralized error handling, user-friendly error messages,
and GPU fallback logic for all headless runners.

This module ensures:
1. Human-readable error messages
2. Actionable suggestions for common errors
3. GPU memory error handling with CPU fallback
4. Clean termination with proper exit codes
"""

import sys
import os
import traceback
from typing import Optional, Callable, Any
from functools import wraps


# ============================================================================
# Error Categories and Messages
# ============================================================================

class ErrorCategory:
    """Error categories for classification."""
    NETWORK = "network"
    FILE_SYSTEM = "file_system"
    MODEL = "model"
    GPU = "gpu"
    AUDIO = "audio"
    CONFIG = "config"
    UNKNOWN = "unknown"


# Common error patterns and their user-friendly messages
ERROR_PATTERNS = {
    # GPU/CUDA errors
    "CUDA out of memory": {
        "category": ErrorCategory.GPU,
        "message": "GPU memory exhausted",
        "suggestion": "Try: (1) Use --cpu flag, (2) Reduce --batch-size, (3) Reduce --segment-size, (4) Close other GPU applications",
        "recoverable": True
    },
    "CUDA error": {
        "category": ErrorCategory.GPU,
        "message": "GPU error occurred",
        "suggestion": "Try using --cpu flag or check GPU driver installation",
        "recoverable": True
    },
    "cuDNN error": {
        "category": ErrorCategory.GPU,
        "message": "cuDNN library error",
        "suggestion": "Reinstall CUDA toolkit or use --cpu flag",
        "recoverable": True
    },
    "DirectML": {
        "category": ErrorCategory.GPU,
        "message": "DirectML error",
        "suggestion": "Check DirectML installation or use --cpu flag",
        "recoverable": True
    },
    
    # Model loading errors
    "No such file or directory": {
        "category": ErrorCategory.FILE_SYSTEM,
        "message": "File not found",
        "suggestion": "Check the file path or use --download to download the model",
        "recoverable": False
    },
    "Permission denied": {
        "category": ErrorCategory.FILE_SYSTEM,
        "message": "Permission denied",
        "suggestion": "Check file/directory permissions or run with appropriate privileges",
        "recoverable": False
    },
    "Invalid model": {
        "category": ErrorCategory.MODEL,
        "message": "Invalid or corrupted model file",
        "suggestion": "Re-download the model using --download flag",
        "recoverable": False
    },
    "Unexpected key": {
        "category": ErrorCategory.MODEL,
        "message": "Model format mismatch",
        "suggestion": "The model file may be corrupted or incompatible. Try re-downloading.",
        "recoverable": False
    },
    "state_dict": {
        "category": ErrorCategory.MODEL,
        "message": "Model checkpoint loading error",
        "suggestion": "The model file may be corrupted. Try re-downloading with --download flag.",
        "recoverable": False
    },
    
    # Audio errors
    "LibsndfileError": {
        "category": ErrorCategory.AUDIO,
        "message": "Audio file read error",
        "suggestion": "Check if the audio file is valid and not corrupted",
        "recoverable": False
    },
    "Unsupported format": {
        "category": ErrorCategory.AUDIO,
        "message": "Unsupported audio format",
        "suggestion": "Convert audio to WAV, FLAC, or MP3 format",
        "recoverable": False
    },
    "Empty audio": {
        "category": ErrorCategory.AUDIO,
        "message": "Audio file is empty or too short",
        "suggestion": "Provide a valid audio file with content",
        "recoverable": False
    },
    
    # Network errors
    "Connection refused": {
        "category": ErrorCategory.NETWORK,
        "message": "Network connection failed",
        "suggestion": "Check your internet connection",
        "recoverable": True
    },
    "timed out": {
        "category": ErrorCategory.NETWORK,
        "message": "Network request timed out",
        "suggestion": "Check your internet connection or try again later",
        "recoverable": True
    },
    "SSL": {
        "category": ErrorCategory.NETWORK,
        "message": "SSL/TLS error",
        "suggestion": "Check your network settings or firewall configuration",
        "recoverable": True
    },
    
    # Config errors
    "KeyError": {
        "category": ErrorCategory.CONFIG,
        "message": "Missing configuration key",
        "suggestion": "Check model configuration file or use --json to provide config",
        "recoverable": False
    },
    "yaml": {
        "category": ErrorCategory.CONFIG,
        "message": "YAML configuration error",
        "suggestion": "Check the YAML config file syntax",
        "recoverable": False
    },
}


def classify_error(error: Exception) -> dict:
    """
    Classify an error and return user-friendly information.
    
    Args:
        error: The exception to classify
        
    Returns:
        Dictionary with category, message, suggestion, and recoverable flag
    """
    error_str = str(error)
    error_type = type(error).__name__
    
    # Check against known patterns
    for pattern, info in ERROR_PATTERNS.items():
        if pattern.lower() in error_str.lower() or pattern.lower() in error_type.lower():
            return {
                "category": info["category"],
                "message": info["message"],
                "suggestion": info["suggestion"],
                "recoverable": info["recoverable"],
                "original": error_str,
                "type": error_type
            }
    
    # Default unknown error
    return {
        "category": ErrorCategory.UNKNOWN,
        "message": f"Unexpected error: {error_type}",
        "suggestion": "Please report this issue with the full error message",
        "recoverable": False,
        "original": error_str,
        "type": error_type
    }


def format_error_message(error_info: dict, verbose: bool = False) -> str:
    """
    Format error information into a user-friendly message.
    
    Args:
        error_info: Dictionary from classify_error()
        verbose: Include technical details
        
    Returns:
        Formatted error message string
    """
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"ERROR: {error_info['message']}")
    lines.append("=" * 60)
    
    if error_info.get('suggestion'):
        lines.append(f"Suggestion: {error_info['suggestion']}")
    
    if verbose and error_info.get('original'):
        lines.append("")
        lines.append("Technical details:")
        lines.append(f"  Type: {error_info.get('type', 'Unknown')}")
        lines.append(f"  Message: {error_info['original'][:200]}")
    
    lines.append("")
    return "\n".join(lines)


def handle_gpu_error(error: Exception, fallback_to_cpu: bool = True) -> bool:
    """
    Handle GPU-related errors with optional CPU fallback.
    
    Args:
        error: The GPU error
        fallback_to_cpu: Whether to suggest CPU fallback
        
    Returns:
        True if error was handled and can retry with CPU
    """
    error_info = classify_error(error)
    
    if error_info["category"] == ErrorCategory.GPU:
        print(format_error_message(error_info, verbose=False), file=sys.stderr)
        
        if fallback_to_cpu and error_info["recoverable"]:
            print("Attempting to fall back to CPU mode...", file=sys.stderr)
            return True
    
    return False


def safe_run(func: Callable, *args, verbose: bool = True, **kwargs) -> tuple:
    """
    Run a function with comprehensive error handling.
    
    Args:
        func: Function to run
        *args: Positional arguments
        verbose: Show verbose error output
        **kwargs: Keyword arguments
        
    Returns:
        Tuple of (success: bool, result_or_error: Any)
    """
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        error_info = classify_error(e)
        print(format_error_message(error_info, verbose=verbose), file=sys.stderr)
        
        if verbose:
            traceback.print_exc()
        
        return False, error_info


def with_error_handling(verbose: bool = True, exit_on_error: bool = True):
    """
    Decorator for functions that need error handling.
    
    Args:
        verbose: Show verbose error output
        exit_on_error: Exit program on error
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                print("\n\nOperation cancelled by user.", file=sys.stderr)
                if exit_on_error:
                    sys.exit(130)  # Standard exit code for Ctrl+C
                raise
            except Exception as e:
                error_info = classify_error(e)
                print(format_error_message(error_info, verbose=verbose), file=sys.stderr)
                
                if verbose:
                    traceback.print_exc()
                
                if exit_on_error:
                    sys.exit(1)
                raise
        return wrapper
    return decorator


class GPUFallbackContext:
    """
    Context manager for GPU operations with automatic CPU fallback.
    
    Usage:
        with GPUFallbackContext() as ctx:
            if ctx.use_gpu:
                # GPU code
            else:
                # CPU code
    """
    
    def __init__(self, prefer_gpu: bool = True, max_retries: int = 1):
        self.prefer_gpu = prefer_gpu
        self.max_retries = max_retries
        self.use_gpu = prefer_gpu
        self.retry_count = 0
        self.last_error = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            error_info = classify_error(exc_val)
            
            if error_info["category"] == ErrorCategory.GPU and self.retry_count < self.max_retries:
                self.retry_count += 1
                self.use_gpu = False
                self.last_error = exc_val
                print(f"\nGPU error occurred. Falling back to CPU (attempt {self.retry_count}/{self.max_retries})...\n", 
                      file=sys.stderr)
                return True  # Suppress exception, allow retry
        
        return False


def check_gpu_availability(verbose: bool = True) -> dict:
    """
    Check GPU availability and return status information.
    
    Returns:
        Dictionary with GPU availability info
    """
    import torch
    
    result = {
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "mps_available": False,
        "directml_available": False,
        "recommended_device": "cpu"
    }
    
    # Check CUDA
    if torch.cuda.is_available():
        result["cuda_available"] = True
        result["cuda_device_count"] = torch.cuda.device_count()
        if result["cuda_device_count"] > 0:
            result["cuda_device_name"] = torch.cuda.get_device_name(0)
            result["recommended_device"] = "cuda"
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        result["mps_available"] = True
        if not result["cuda_available"]:
            result["recommended_device"] = "mps"
    
    # Check DirectML (basic check)
    try:
        import torch_directml
        result["directml_available"] = True
        if not result["cuda_available"] and not result["mps_available"]:
            result["recommended_device"] = "directml"
    except ImportError:
        pass
    
    if verbose:
        print("GPU Status:")
        print(f"  CUDA: {'Available' if result['cuda_available'] else 'Not available'}")
        if result["cuda_available"]:
            print(f"    Devices: {result['cuda_device_count']}")
            print(f"    Name: {result['cuda_device_name']}")
        print(f"  MPS (Apple): {'Available' if result['mps_available'] else 'Not available'}")
        print(f"  DirectML: {'Available' if result['directml_available'] else 'Not available'}")
        print(f"  Recommended: {result['recommended_device']}")
    
    return result


def validate_audio_file(filepath: str) -> tuple:
    """
    Validate an audio file before processing.
    
    Args:
        filepath: Path to audio file
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not os.path.exists(filepath):
        return False, f"Audio file not found: {filepath}"
    
    if not os.path.isfile(filepath):
        return False, f"Path is not a file: {filepath}"
    
    size = os.path.getsize(filepath)
    if size == 0:
        return False, f"Audio file is empty: {filepath}"
    
    if size < 1024:  # Less than 1KB
        return False, f"Audio file too small ({size} bytes): {filepath}"
    
    # Check extension
    valid_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.aiff'}
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in valid_extensions:
        return False, f"Unsupported audio format '{ext}'. Supported: {', '.join(valid_extensions)}"
    
    return True, "Audio file validated"


def validate_output_directory(dirpath: str) -> tuple:
    """
    Validate and prepare output directory.
    
    Args:
        dirpath: Path to output directory
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    try:
        if os.path.exists(dirpath):
            if not os.path.isdir(dirpath):
                return False, f"Output path exists but is not a directory: {dirpath}"
            if not os.access(dirpath, os.W_OK):
                return False, f"No write permission for output directory: {dirpath}"
        else:
            # Try to create directory
            os.makedirs(dirpath, exist_ok=True)
        
        return True, "Output directory ready"
    except OSError as e:
        return False, f"Cannot create output directory: {e}"
