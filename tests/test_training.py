import itertools
import os.path
import pathlib
import shutil
import threading
import time

# ---------------------------------------------------------------------
# Timer bar helper
# ---------------------------------------------------------------------
from contextlib import contextmanager
from typing import Dict, List

import pytest
import torch
from hydra import compose, initialize
from tqdm import tqdm

from controllable_patching_striding.train import CONFIG_DIR, CONFIG_NAME, main
from controllable_patching_striding.trainer.checkpoints import (
    CHECKPOINT_METADA_FILENAME,
)


@contextmanager
def timebar(desc="⏳ Running"):
    """
    Displays a live elapsed-time bar while the wrapped block executes.
    Does not affect pytest output formatting.
    """
    stop = False

    def ticker():
        with tqdm(bar_format="{desc}: {elapsed}", desc=desc, leave=False) as t:
            while not stop:
                time.sleep(0.1)
                t.update(0)

    thr = threading.Thread(target=ticker, daemon=True)
    thr.start()
    try:
        yield
    finally:
        stop = True
        thr.join(timeout=1)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def default_conf(dummy_dataset, checkpoint_folder) -> Dict[str, str | bool]:
    """Generate default training options as a dictionary."""
    return {
        "server": "local",
        "logger": "none",
        "trainer": "debug",
        "data": "debug",
        "model": "debug",
        "data.well_base_path": dummy_dataset,
        "+data.module_parameters.well_dataset_info.dummy.path": dummy_dataset / "dummy",
        "data_workers": "1",
        "name": "test",
        "checkpoint.save_dir": checkpoint_folder,
        "automatic_setup": False,
    }


def format_overrides(overrides: Dict[str, str]) -> List[str]:
    """Format training options from dictionary to list of overrides."""
    return [f"{key}={val}" for key, val in overrides.items()]


@pytest.fixture()
def conf(default_conf, request):
    """Generate overrides by combining default configuration and the various test options."""
    override_dict = default_conf
    if hasattr(request, "param"):
        override_dict.update(request.param)
    overrides = format_overrides(override_dict)
    return overrides


@pytest.fixture()
def checkpoint_folder(tmp_path):
    """Create and clean a temporary folder for checkpoints"""
    yield tmp_path / "checkpoints"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)


def generate_parameters(conf_options: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """Generate all the possible combination of options."""
    conf_combinations = []
    for combination in itertools.product(*conf_options.values()):
        conf_combinations.append(dict(zip(conf_options.keys(), combination)))
    return conf_combinations


# ---------------------------------------------------------------------
# Test the parameter set used in the paper. TODO: Add more tests.
# ---------------------------------------------------------------------

conf_options = {
    "trainer.prediction_type": ["delta"],
    "trainer.enable_amp": ["False"],
    "model.causal_in_time": ["True"],
}


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

@pytest.mark.parametrize("conf", generate_parameters(conf_options), indirect=True)
def test_train(conf):
    """Test training terminates normally for different sets of config."""
    overrides = conf
    cfg_dir = os.path.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)

    with timebar("⏳ test_train"):
        with initialize(config_path=str(cfg_dir)):
            cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
            with torch.autograd.detect_anomaly():
                main(cfg)

    assert True


def test_checkpoints(checkpoint_folder, conf):
    """Test training generates checkpoints as expected."""
    overrides = conf
    overrides += ["checkpoint.checkpoint_frequency=1", "trainer.max_epoch=2"]
    cfg_dir = os.path.relpath(CONFIG_DIR, os.path.dirname(__file__))

    with timebar("⏳ test_checkpoints"):
        with initialize(config_path=cfg_dir):
            cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
            main(cfg)

    # Check epoch 1 checkpoint exists
    assert os.path.isdir(checkpoint_folder / "step_1")
    # Check best checkpoint exists
    assert os.path.islink(checkpoint_folder / "best")
    assert len(os.listdir(checkpoint_folder / "best")) > 0
    # Check metadata have been saved
    assert CHECKPOINT_METADA_FILENAME in os.listdir(checkpoint_folder / "best")
    # Check last checkpoint exists and is symlink of step_2
    assert os.path.islink(checkpoint_folder / "last")
    assert os.path.samefile(
        checkpoint_folder / "last", checkpoint_folder / "step_2"
    )


def test_no_checkpoint(checkpoint_folder, conf):
    """Test no checkpoints are saved when expected."""
    overrides = conf
    overrides += ["checkpoint=none"]
    cfg_dir = os.path.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)

    with timebar("⏳ test_no_checkpoint"):
        with initialize(config_path=cfg_dir):
            cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
            main(cfg)

    # Checkpoint folder should not exist
    assert not os.path.exists(checkpoint_folder)


def test_resume_from_checkpoint(checkpoint_folder, conf):
    """Test in a circumventing way training can resume from checkpoint"""
    overrides = conf
    overrides += [
        "checkpoint.checkpoint_frequency=0",
    ]
    cfg_dir = os.path.relpath(CONFIG_DIR, os.path.dirname(__file__))
    assert os.path.exists(checkpoint_folder) is False

    # First run
    with timebar("⏳ test_resume_from_checkpoint (run 1)"):
        with initialize(config_path=cfg_dir):
            cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
            main(cfg)

    assert "step_1" in os.listdir(checkpoint_folder)

    # Resume run
    overrides += ["checkpoint.checkpoint_frequency=1", "trainer.max_epoch=2"]

    with timebar("⏳ test_resume_from_checkpoint (run 2)"):
        with initialize(config_path=cfg_dir):
            cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
            main(cfg)

    assert "step_2" in os.listdir(checkpoint_folder)
    assert os.path.samefile(checkpoint_folder / "last", checkpoint_folder / "step_2")
    assert not os.path.samefile(
        checkpoint_folder / "step_1", checkpoint_folder / "step_2"
    )
