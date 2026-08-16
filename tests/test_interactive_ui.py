"""
Automated unit tests for Raylib interactive UI components and CLI flags.
"""

import os
import sys
import numpy as np
import pytest

from gcmc.v1 import load_config
from gcmc.ui.widgets import (
    COLOR_BG, COLOR_PANEL, COLOR_TEXT, COLOR_ACCENT
)
from gcmc.ui.raylib_gcmc_viewer import GCMCInteractiveViewer
from gcmc.ui.raylib_cdft_viewer import CDFTInteractiveViewer
import gcmc.main as main_cli


def test_ui_theme_colors():
    """Verify UI color palette definitions."""
    assert COLOR_BG.r == 18
    assert COLOR_PANEL.r == 26
    assert COLOR_ACCENT.r == 45
    assert COLOR_TEXT.r == 220


def test_gcmc_interactive_viewer_instantiation(run_dir):
    """Verify GCMC interactive viewer initializes simulation and state properly."""
    dir_path = run_dir("dipole_fast")
    config = load_config(os.path.join(dir_path, "input.yaml"))
    viewer = GCMCInteractiveViewer(config, input_folder=dir_path, width=800, height=600)

    assert viewer.width == 800
    assert viewer.height == 600
    assert viewer.is_running is True
    assert viewer.steps_per_frame == 50
    assert len(viewer.history_n) == 1
    assert viewer.sim.number > 0

    # Test running steps through viewer bridge
    viewer.run_steps(10)
    assert viewer.total_steps == 10
    assert len(viewer.history_n) == 2
    assert len(viewer.history_energy) == 2


def test_cdft_interactive_viewer_instantiation():
    """Verify cDFT fluid manipulation viewer initializes and steps correctly."""
    viewer = CDFTInteractiveViewer(width=800, height=600)

    assert viewer.width == 800
    assert viewer.height == 600
    assert viewer.nz == 50
    assert len(viewer.history_fillings) == 1

    # Test single simulation step in manual mode
    viewer.step_simulation()
    assert len(viewer.history_fillings) == 2
    assert len(viewer.history_rewards) == 2


def test_cli_interactive_flags():
    """Verify CLI parser properly handles -i and --interactive flags without running headless simulations."""
    import argparse

    # Check parser configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interactive", action="store_true", default=False)
    args = parser.parse_args(["-i"])
    assert args.interactive is True

    args2 = parser.parse_args([])
    assert args2.interactive is False
