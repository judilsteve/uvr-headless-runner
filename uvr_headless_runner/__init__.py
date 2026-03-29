"""
UVR Headless Runner - Audio source separation library

Expose main runner functions for programmatic use.
"""

from uvr_headless_runner.demucs_headless_runner import run_demucs_headless
from uvr_headless_runner.vr_headless_runner import run_vr_headless
from uvr_headless_runner.mdx_headless_runner import run_mdx_headless

__version__ = "1.1.0"

__all__ = [
    "run_demucs_headless",
    "run_vr_headless", 
    "run_mdx_headless",
]
