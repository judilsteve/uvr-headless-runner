#!/usr/bin/env python3
"""
CLI Progress System for UVR Headless Runners
=============================================

A professional CLI progress reporting system that matches or exceeds 
the UX of UVR GUI for terminal-based audio processing workflows.

Features:
- Real-time progress bars for model downloads, loading, and inference
- Speed and ETA display
- Stage-based progress tracking
- Docker-compatible output
- Graceful fallback when rich/tqdm unavailable

Usage:
    from progress import ProgressManager, ProgressStage
    
    with ProgressManager(verbose=True) as pm:
        pm.start_stage(ProgressStage.DOWNLOADING_MODEL)
        pm.update_progress(current=50, total=100)
        pm.finish_stage()
"""

import os
import sys
import time
from enum import Enum, auto
from typing import Optional, Callable, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass, field

# Try to import rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn, 
        TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn,
        DownloadColumn, TransferSpeedColumn, MofNCompleteColumn
    )
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.live import Live
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Fallback to tqdm if rich not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class ProgressStage(Enum):
    """Processing stages for audio separation."""
    INITIALIZING = auto()
    DOWNLOADING_MODEL = auto()
    LOADING_MODEL = auto()
    LOADING_AUDIO = auto()
    PROCESSING_AUDIO = auto()
    INFERENCE = auto()
    POST_PROCESSING = auto()
    SAVING_OUTPUTS = auto()
    COMPLETE = auto()
    ERROR = auto()


# Stage descriptions for display
STAGE_DESCRIPTIONS = {
    ProgressStage.INITIALIZING: "Initializing",
    ProgressStage.DOWNLOADING_MODEL: "Downloading model",
    ProgressStage.LOADING_MODEL: "Loading model",
    ProgressStage.LOADING_AUDIO: "Loading audio",
    ProgressStage.PROCESSING_AUDIO: "Processing audio",
    ProgressStage.INFERENCE: "Running inference",
    ProgressStage.POST_PROCESSING: "Post-processing",
    ProgressStage.SAVING_OUTPUTS: "Saving outputs",
    ProgressStage.COMPLETE: "Complete",
    ProgressStage.ERROR: "Error",
}


# Colors for different stages
STAGE_COLORS = {
    ProgressStage.INITIALIZING: "cyan",
    ProgressStage.DOWNLOADING_MODEL: "blue",
    ProgressStage.LOADING_MODEL: "yellow",
    ProgressStage.LOADING_AUDIO: "yellow",
    ProgressStage.PROCESSING_AUDIO: "green",
    ProgressStage.INFERENCE: "green",
    ProgressStage.POST_PROCESSING: "magenta",
    ProgressStage.SAVING_OUTPUTS: "cyan",
    ProgressStage.COMPLETE: "green",
    ProgressStage.ERROR: "red",
}


@dataclass
class StageProgress:
    """Track progress within a stage."""
    current: int = 0
    total: int = 100
    start_time: float = field(default_factory=time.time)
    description: str = ""
    
    @property
    def percentage(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(100.0, (self.current / self.total) * 100)
    
    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    @property
    def eta(self) -> Optional[float]:
        if self.current <= 0 or self.percentage <= 0:
            return None
        return (self.elapsed / self.percentage) * (100 - self.percentage)
    
    @property
    def speed(self) -> float:
        """Items per second."""
        if self.elapsed <= 0:
            return 0.0
        return self.current / self.elapsed


def format_time(seconds: Optional[float]) -> str:
    """Format seconds to human-readable string."""
    if seconds is None or seconds < 0:
        return "--:--"
    
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}:{mins:02d}:00"


def format_bytes(size: float) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


class BaseProgressHandler:
    """Base class for progress handlers."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.current_stage: Optional[ProgressStage] = None
        self.stage_progress = StageProgress()
        self._model_name: str = ""
        self._file_name: str = ""
    
    def set_model_name(self, name: str):
        self._model_name = name
    
    def set_file_name(self, name: str):
        self._file_name = name
    
    def start_stage(self, stage: ProgressStage, description: str = "", total: int = 100):
        """Start a new processing stage."""
        self.current_stage = stage
        self.stage_progress = StageProgress(
            current=0,
            total=total,
            start_time=time.time(),
            description=description or STAGE_DESCRIPTIONS.get(stage, "")
        )
        self._on_stage_start()
    
    def update_progress(self, current: int = None, total: int = None, 
                        description: str = None, increment: int = None):
        """Update progress within current stage."""
        if current is not None:
            self.stage_progress.current = current
        elif increment is not None:
            self.stage_progress.current += increment
        
        if total is not None:
            self.stage_progress.total = total
        
        if description is not None:
            self.stage_progress.description = description
        
        self._on_progress_update()
    
    def finish_stage(self, message: str = ""):
        """Finish the current stage."""
        if self.stage_progress:
            self.stage_progress.current = self.stage_progress.total
        self._on_stage_finish(message)
    
    def write_message(self, message: str, style: str = ""):
        """Write a message to output."""
        self._on_message(message, style)
    
    def _on_stage_start(self):
        raise NotImplementedError
    
    def _on_progress_update(self):
        raise NotImplementedError
    
    def _on_stage_finish(self, message: str):
        raise NotImplementedError
    
    def _on_message(self, message: str, style: str):
        raise NotImplementedError
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Only define RichProgressHandler if rich is available
if RICH_AVAILABLE:
    class RichProgressHandler(BaseProgressHandler):
        """Rich-based progress handler with beautiful terminal output."""
        
        def __init__(self, verbose: bool = True):
            super().__init__(verbose)
            self.console = Console()
            self.progress: Optional[Progress] = None
            self.task_id = None
            self._live = None
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.progress:
                self.progress.stop()
            if self._live:
                self._live.stop()
        
        def _create_progress(self, is_download: bool = False) -> Progress:
            """Create appropriate progress bar based on stage type."""
            if is_download:
                return Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(bar_width=40),
                    TaskProgressColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                    console=self.console,
                    transient=False,
                )
            else:
                return Progress(
                    SpinnerColumn(),
                    TextColumn("[bold green]{task.description}"),
                    BarColumn(bar_width=40, complete_style="green", finished_style="bright_green"),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    TextColumn("•"),
                    TimeRemainingColumn(),
                    console=self.console,
                    transient=False,
                )
        
        def _on_stage_start(self):
            if not self.verbose:
                return
            
            # Stop previous progress if any
            if self.progress:
                self.progress.stop()
            
            stage = self.current_stage
            is_download = stage == ProgressStage.DOWNLOADING_MODEL
            
            self.progress = self._create_progress(is_download)
            
            description = self.stage_progress.description
            if self._model_name and stage in [ProgressStage.LOADING_MODEL, ProgressStage.DOWNLOADING_MODEL]:
                description = f"{description}: {self._model_name}"
            elif self._file_name and stage in [ProgressStage.LOADING_AUDIO, ProgressStage.SAVING_OUTPUTS]:
                description = f"{description}: {self._file_name}"
            
            self.progress.start()
            self.task_id = self.progress.add_task(
                description,
                total=self.stage_progress.total
            )
        
        def _on_progress_update(self):
            if not self.verbose or not self.progress or self.task_id is None:
                return
            
            self.progress.update(
                self.task_id,
                completed=self.stage_progress.current,
                total=self.stage_progress.total,
                description=self.stage_progress.description
            )
        
        def _on_stage_finish(self, message: str):
            if not self.verbose:
                return
            
            if self.progress and self.task_id is not None:
                self.progress.update(
                    self.task_id,
                    completed=self.stage_progress.total
                )
                self.progress.stop()
                self.progress = None
                self.task_id = None
            
            if message:
                stage = self.current_stage
                color = STAGE_COLORS.get(stage, "white")
                self.console.print(f"[{color}]✓[/{color}] {message}")
        
        def _on_message(self, message: str, style: str):
            if not self.verbose:
                return
            
            if style:
                self.console.print(f"[{style}]{message}[/{style}]")
            else:
                self.console.print(message)
        
        def print_header(self, model_name: str, input_file: str, output_path: str, 
                        device: str, arch_type: str, output_stems: str = None):
            """Print a formatted header for the processing job."""
            if not self.verbose:
                return
            
            table = Table(show_header=False, box=box.ROUNDED, 
                         border_style="cyan", padding=(0, 1))
            table.add_column("Key", style="bold cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Model", model_name)
            table.add_row("Input", os.path.basename(input_file))
            table.add_row("Output", output_path)
            if output_stems:
                table.add_row("Stems", output_stems)
            table.add_row("Device", device)
            table.add_row("Architecture", arch_type)
            
            self.console.print()
            self.console.print(Panel(table, title="[bold]UVR Audio Separation[/bold]", 
                                     border_style="blue"))
            self.console.print()
        
        def print_summary(self, elapsed_time: float, output_files: list):
            """Print a summary of the completed processing."""
            if not self.verbose:
                return
            
            self.console.print()
            self.console.print(Panel(
                f"[green]✓ Processing completed in {format_time(elapsed_time)}[/green]",
                border_style="green"
            ))
            
            if output_files:
                self.console.print("\n[bold]Output files:[/bold]")
                for f in output_files:
                    self.console.print(f"  • {f}")
            self.console.print()


# Only define TqdmProgressHandler if tqdm is available
if TQDM_AVAILABLE:
    class TqdmProgressHandler(BaseProgressHandler):
        """TQDM-based progress handler fallback."""
        
        def __init__(self, verbose: bool = True):
            super().__init__(verbose)
            self.pbar: Optional[tqdm] = None
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.pbar:
                self.pbar.close()
        
        def _on_stage_start(self):
            if not self.verbose:
                return
            
            if self.pbar:
                self.pbar.close()
            
            description = self.stage_progress.description
            if self._model_name and self.current_stage in [ProgressStage.LOADING_MODEL, ProgressStage.DOWNLOADING_MODEL]:
                description = f"{description}: {self._model_name}"
            
            unit = "B" if self.current_stage == ProgressStage.DOWNLOADING_MODEL else "it"
            unit_scale = self.current_stage == ProgressStage.DOWNLOADING_MODEL
            
            self.pbar = tqdm(
                total=self.stage_progress.total,
                desc=description,
                unit=unit,
                unit_scale=unit_scale,
                ncols=80,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        
        def _on_progress_update(self):
            if not self.verbose or not self.pbar:
                return
            
            # Update to current value
            diff = self.stage_progress.current - self.pbar.n
            if diff > 0:
                self.pbar.update(diff)
            
            if self.stage_progress.description != self.pbar.desc:
                self.pbar.set_description(self.stage_progress.description)
        
        def _on_stage_finish(self, message: str):
            if not self.verbose:
                return
            
            if self.pbar:
                self.pbar.close()
                self.pbar = None
            
            if message:
                print(f"✓ {message}")
        
        def _on_message(self, message: str, style: str):
            if not self.verbose:
                return
            tqdm.write(message)
        
        def print_header(self, model_name: str, input_file: str, output_path: str,
                        device: str, arch_type: str, output_stems: str = None):
            if not self.verbose:
                return
            
            print("=" * 60)
            print("UVR Audio Separation")
            print("=" * 60)
            print(f"Model:        {model_name}")
            print(f"Input:        {os.path.basename(input_file)}")
            print(f"Output:       {output_path}")
            if output_stems:
                print(f"Stems:        {output_stems}")
            print(f"Device:       {device}")
            print(f"Architecture: {arch_type}")
            print("=" * 60)
        
        def print_summary(self, elapsed_time: float, output_files: list):
            if not self.verbose:
                return
            
            print()
            print(f"[OK] Processing completed in {format_time(elapsed_time)}")
            if output_files:
                print("\nOutput files:")
                for f in output_files:
                    print(f"  - {f}")
            print()


class BasicProgressHandler(BaseProgressHandler):
    """Basic text-based progress handler (no dependencies)."""
    
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self._last_print_time = 0
        self._print_interval = 0.5  # Update every 0.5 seconds
    
    def _on_stage_start(self):
        if not self.verbose:
            return
        
        description = self.stage_progress.description
        print(f"\n[{description}]")
        sys.stdout.flush()
    
    def _on_progress_update(self):
        if not self.verbose:
            return
        
        # Rate limit output
        now = time.time()
        if now - self._last_print_time < self._print_interval:
            return
        self._last_print_time = now
        
        pct = self.stage_progress.percentage
        eta = self.stage_progress.eta
        eta_str = f"ETA: {format_time(eta)}" if eta else ""
        
        # Simple text progress bar
        bar_width = 30
        filled = int(bar_width * pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Print with carriage return for in-place update
        print(f"\r  [{bar}] {pct:5.1f}% {eta_str}    ", end="", flush=True)
    
    def _on_stage_finish(self, message: str):
        if not self.verbose:
            return
        
        # Clear the progress line
        print("\r" + " " * 60 + "\r", end="")
        
        if message:
            print(f"✓ {message}")
        else:
            print(f"✓ {self.stage_progress.description} completed")
        sys.stdout.flush()
    
    def _on_message(self, message: str, style: str):
        if not self.verbose:
            return
        print(message)
        sys.stdout.flush()
    
    def print_header(self, model_name: str, input_file: str, output_path: str,
                    device: str, arch_type: str, output_stems: str = None):
        if not self.verbose:
            return
        
        print("=" * 60)
        print("UVR Audio Separation")
        print("=" * 60)
        print(f"Model:        {model_name}")
        print(f"Input:        {os.path.basename(input_file)}")
        print(f"Output:       {output_path}")
        if output_stems:
            print(f"Stems:        {output_stems}")
        print(f"Device:       {device}")
        print(f"Architecture: {arch_type}")
        print("=" * 60)
    
    def print_summary(self, elapsed_time: float, output_files: list):
        if not self.verbose:
            return
        
        print()
        print(f"[OK] Processing completed in {format_time(elapsed_time)}")
        if output_files:
            print("\nOutput files:")
            for f in output_files:
                print(f"  • {f}")
        print()


class ProgressManager:
    """
    Main progress manager that automatically selects the best available handler.
    
    Usage:
        with ProgressManager(verbose=True) as pm:
            pm.print_header(model_name, input_file, output_path, device, arch_type)
            
            pm.start_stage(ProgressStage.LOADING_MODEL)
            # ... model loading ...
            pm.finish_stage("Model loaded")
            
            pm.start_stage(ProgressStage.INFERENCE, total=100)
            for i in range(100):
                # ... processing ...
                pm.update_progress(current=i+1)
            pm.finish_stage("Inference complete")
    """
    
    def __init__(self, verbose: bool = True, prefer_rich: bool = True):
        """
        Initialize progress manager.
        
        Args:
            verbose: Whether to show progress output
            prefer_rich: Whether to prefer rich library if available
        """
        self.verbose = verbose
        self.start_time = time.time()
        self._output_files: list = []
        
        # Select best available handler
        if verbose and prefer_rich and RICH_AVAILABLE:
            self.handler = RichProgressHandler(verbose)
        elif verbose and TQDM_AVAILABLE:
            self.handler = TqdmProgressHandler(verbose)
        else:
            self.handler = BasicProgressHandler(verbose)
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.handler, '__exit__'):
            self.handler.__exit__(exc_type, exc_val, exc_tb)
        
        if exc_type is None and self.verbose:
            elapsed = time.time() - self.start_time
            self.handler.print_summary(elapsed, self._output_files)
    
    def set_model_name(self, name: str):
        """Set the current model name for display."""
        self.handler.set_model_name(name)
    
    def set_file_name(self, name: str):
        """Set the current file name for display."""
        self.handler.set_file_name(name)
    
    def add_output_file(self, path: str):
        """Register an output file for the summary."""
        self._output_files.append(path)
    
    def print_header(self, model_name: str, input_file: str, output_path: str,
                    device: str, arch_type: str, output_stems: str = None):
        """Print processing header information."""
        self.handler.print_header(model_name, input_file, output_path, device, arch_type, output_stems)
    
    def start_stage(self, stage: ProgressStage, description: str = "", total: int = 100):
        """Start a new processing stage."""
        self.handler.start_stage(stage, description, total)
    
    def update_progress(self, current: int = None, total: int = None,
                       description: str = None, increment: int = None):
        """Update progress within current stage."""
        self.handler.update_progress(current, total, description, increment)
    
    def finish_stage(self, message: str = ""):
        """Finish the current stage."""
        self.handler.finish_stage(message)
    
    def write_message(self, message: str, style: str = ""):
        """Write a message to output."""
        self.handler.write_message(message, style)


# ============================================================================
# UVR Callback Integration Functions - PARITY IMPLEMENTATION
# ============================================================================
# 
# GUI Progress Formula Analysis (from UVR.py):
# =============================================
# 
# For single file, single model processing (CLI case):
#   total_count = true_model_count * total_files = 1 * 1 = 1
#   base = 100 / total_count = 100
#   progress = base * iteration - base + base * step
#            = 100 * (iteration - 1 + step)
#            = 100 * step  (since iteration = 1 for first file)
# 
# Therefore: GUI_progress% = step * 100
# 
# The UVR separator code calls set_progress_bar with:
#   step = 0.05                               -> 5%   (initial setup done)
#   step = 0.1 + inference_iterations         -> 10% to 90% (during inference)
#         where inference_iterations = 0.8 / length * progress_value
#   step = 0.95                               -> 95%  (saving outputs)
# 
# During inference:
#   GUI_progress = (0.1 + 0.8/length * progress_value) * 100
#                = 10 + 80 * (progress_value / length)
# 
# This implementation mirrors this formula EXACTLY.
# ============================================================================

def create_progress_callbacks(progress_manager: ProgressManager, total_iterations: int = 100):
    """
    Create callback functions compatible with UVR's process_data structure.
    
    PARITY GUARANTEE:
    This creates set_progress_bar and write_to_console callbacks that produce
    IDENTICAL progress percentages to UVR GUI for single-file processing.
    
    GUI Formula (single file, single model):
        GUI_progress% = step * 100
        where step = 0.1 + (0.8/length * progress_value) during inference
    
    Args:
        progress_manager: ProgressManager instance
        total_iterations: Expected total iterations (used for progress bar total)
        
    Returns:
        Dictionary with progress callback functions
    """
    # State tracking (use lists for mutability in closure)
    _state = {
        'inference_started': False,
        'last_gui_pct': 0.0,
        'stage_active': False,
    }
    
    def set_progress_bar(step: float = 0, inference_iterations: float = 0):
        """
        UVR-compatible progress callback with EXACT GUI parity.
        
        UVR GUI Formula:
            progress% = (step + inference_iterations) * 100
            
        Where:
            - step = 0.05: Initial setup complete (5%)
            - step = 0.1, inference_iterations = 0: Start inference (10%)
            - step = 0.1, inference_iterations = 0.8*i/n: During inference (10%-90%)
            - step = 0.95: Saving outputs (95%)
            
        The inference_iterations parameter contains the value (0.8/length * progress_value),
        so the total step during inference is 0.1 + inference_iterations.
        """
        # Calculate GUI-equivalent percentage
        # In UVR GUI: progress = step * 100 (for single file)
        # The 'step' passed to us is the base step (0.05, 0.1, 0.95)
        # The 'inference_iterations' is added to step for inference progress
        gui_pct = (step + inference_iterations) * 100
        gui_pct = min(gui_pct, 100.0)  # Cap at 100%
        
        # Track for logging
        _state['last_gui_pct'] = gui_pct
        
        # Stage transitions based on progress thresholds
        if gui_pct <= 5.0:
            # Initial setup done (5%)
            if not _state['stage_active']:
                progress_manager.start_stage(ProgressStage.LOADING_AUDIO, total=100)
                _state['stage_active'] = True
            progress_manager.update_progress(current=int(gui_pct))
            
        elif gui_pct > 5.0 and gui_pct <= 10.0 and not _state['inference_started']:
            # Transition to inference stage (5% -> 10%)
            if _state['stage_active']:
                progress_manager.finish_stage("Audio loaded")
            progress_manager.start_stage(ProgressStage.INFERENCE, total=100)
            _state['inference_started'] = True
            _state['stage_active'] = True
            progress_manager.update_progress(current=int(gui_pct))
            
        elif gui_pct > 10.0 and gui_pct < 95.0:
            # During inference (10% -> 90%)
            if not _state['inference_started']:
                # Late start - jumped straight to inference
                progress_manager.start_stage(ProgressStage.INFERENCE, total=100)
                _state['inference_started'] = True
                _state['stage_active'] = True
            progress_manager.update_progress(current=int(gui_pct))
            
        elif gui_pct >= 95.0:
            # Saving outputs (95%+)
            if _state['inference_started'] and _state['stage_active']:
                progress_manager.finish_stage("Inference complete")
                _state['stage_active'] = False
            if not _state['stage_active']:
                progress_manager.start_stage(ProgressStage.SAVING_OUTPUTS, total=100)
                _state['stage_active'] = True
            progress_manager.update_progress(current=int(gui_pct))
    
    def write_to_console(progress_text: str = '', base_text: str = ''):
        """UVR-compatible console write callback."""
        message = f"{base_text}{progress_text}".strip()
        if message and progress_manager.verbose:
            # Filter out common UVR noise messages and redundant stem info
            # (stem info is now shown in header)
            noise_patterns = ['Done!', 'DONE', 'done!', 'Saving ', ' stem...']
            if not any(p in message for p in noise_patterns):
                progress_manager.write_message(message)
    
    def process_iteration():
        """Called when processing moves to next iteration (ensemble mode)."""
        pass  # Not used in single-file CLI processing
    
    return {
        'set_progress_bar': set_progress_bar,
        'write_to_console': write_to_console,
        'process_iteration': process_iteration,
    }


def calculate_gui_progress(step: float, inference_iterations: float = 0) -> float:
    """
    Calculate exact GUI progress percentage.
    
    This is the reference formula extracted from UVR.py:
        GUI_progress% = (step + inference_iterations) * 100
    
    For inference progress:
        inference_iterations = 0.8 / length * progress_value
    
    So during inference:
        GUI_progress% = (0.1 + 0.8/length * progress_value) * 100
                      = 10 + 80 * (progress_value / length)
    
    Args:
        step: Base step value (0.05, 0.1, 0.95, etc.)
        inference_iterations: Additional inference progress (0.8/length * progress_value)
        
    Returns:
        GUI progress percentage (0-100)
    """
    return min((step + inference_iterations) * 100, 100.0)


def create_download_progress_callback(progress_manager: ProgressManager):
    """
    Create a download progress callback for model_downloader.py integration.
    
    Args:
        progress_manager: ProgressManager instance
        
    Returns:
        Callback function(current_bytes, total_bytes, filename)
    """
    _started = [False]
    
    def progress_callback(current: int, total: int, filename: str):
        if not _started[0]:
            progress_manager.start_stage(
                ProgressStage.DOWNLOADING_MODEL,
                description=f"Downloading: {filename}",
                total=total
            )
            _started[0] = True
        
        progress_manager.update_progress(current=current, total=total)
        
        if current >= total:
            progress_manager.finish_stage(f"Downloaded: {filename}")
            _started[0] = False
    
    return progress_callback


# ============================================================================
# Inference Progress Tracker
# ============================================================================

class InferenceProgressTracker:
    """
    Tracks inference progress across different model architectures.
    
    This class provides a unified interface for tracking progress in
    MDX, MDX-C/Roformer, Demucs, and VR architectures.
    """
    
    def __init__(self, progress_manager: ProgressManager, arch_type: str = "MDX"):
        self.pm = progress_manager
        self.arch_type = arch_type
        self.total_steps = 0
        self.current_step = 0
        self.start_time = time.time()
    
    def set_total_steps(self, total: int):
        """Set the total number of inference steps."""
        self.total_steps = max(1, total)
        self.current_step = 0
    
    def step(self, description: str = ""):
        """Advance one step and update progress."""
        self.current_step += 1
        self.pm.update_progress(
            current=min(self.current_step, self.total_steps),
            total=self.total_steps,
            description=description if description else None
        )
    
    def get_callback(self):
        """
        Get a callback function for use in inference loops.
        
        Returns:
            Function that advances progress by one step when called.
        """
        def callback():
            self.step()
        return callback


# ============================================================================
# Utility Functions
# ============================================================================

def is_docker_environment() -> bool:
    """Check if running inside Docker container."""
    # Check for .dockerenv file
    if os.path.exists('/.dockerenv'):
        return True
    
    # Check cgroup
    try:
        with open('/proc/1/cgroup', 'rt') as f:
            return 'docker' in f.read()
    except:
        pass
    
    return False


def get_terminal_width() -> int:
    """Get terminal width, with fallback for non-terminal environments."""
    try:
        import shutil
        return shutil.get_terminal_size().columns
    except:
        return 80


def check_progress_dependencies() -> Dict[str, bool]:
    """Check which progress display libraries are available."""
    return {
        'rich': RICH_AVAILABLE,
        'tqdm': TQDM_AVAILABLE,
        'in_docker': is_docker_environment(),
        'terminal_width': get_terminal_width(),
    }


# ============================================================================
# CLI Demo
# ============================================================================

def demo():
    """Demonstrate the progress system."""
    import time
    
    print("Progress System Demo")
    print("=" * 60)
    
    deps = check_progress_dependencies()
    print(f"Rich available: {deps['rich']}")
    print(f"TQDM available: {deps['tqdm']}")
    print(f"In Docker: {deps['in_docker']}")
    print()
    
    with ProgressManager(verbose=True) as pm:
        pm.print_header(
            model_name="UVR-MDX-NET Inst HQ 3",
            input_file="test_song.mp3",
            output_path="./output",
            device="CUDA:0",
            arch_type="MDX-Net",
            output_stems="Vocals, Instrumental"
        )
        
        # Simulate download
        pm.start_stage(ProgressStage.DOWNLOADING_MODEL, total=100)
        for i in range(100):
            time.sleep(0.02)
            pm.update_progress(current=i+1)
        pm.finish_stage("Model downloaded")
        
        # Simulate loading
        pm.start_stage(ProgressStage.LOADING_MODEL)
        time.sleep(0.5)
        pm.finish_stage("Model loaded")
        
        # Simulate inference
        pm.start_stage(ProgressStage.INFERENCE, total=50)
        for i in range(50):
            time.sleep(0.05)
            pm.update_progress(current=i+1)
        pm.finish_stage("Inference complete")
        
        # Simulate saving
        pm.start_stage(ProgressStage.SAVING_OUTPUTS)
        time.sleep(0.3)
        pm.add_output_file("output/test_song_(Vocals).wav")
        pm.add_output_file("output/test_song_(Instrumental).wav")
        pm.finish_stage("Outputs saved")


if __name__ == '__main__':
    demo()
