"""Pytest fixtures for GCMC v1 tests."""

import os
import shutil

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(TESTS_DIR, "test_configs")


@pytest.fixture
def run_dir(tmp_path):
    """
    Creates a temporary directory for running a test with copied inputs.
    """

    def _prepare(config_name):
        src_dir = os.path.join(CONFIGS_DIR, config_name)
        dest_dir = tmp_path / config_name
        shutil.copytree(src_dir, dest_dir)
        return str(dest_dir)

    return _prepare
