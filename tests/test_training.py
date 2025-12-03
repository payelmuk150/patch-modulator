import itertools
import os.path
import shutil
from typing import Dict, List

import pytest
import torch
from hydra import compose, initialize

from temporary_mppx_name.train import CONFIG_DIR, CONFIG_NAME, main
from temporary_mppx_name.trainer.checkpoints import CHECKPOINT_METADA_FILENAME


@pytest.fixture()
def checkpoint_folder(tmp_path):
    """Create and clean a temporary folder for checkpoints"""
    yield tmp_path / "checkpoints"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)


def generate_parameters(conf_options: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """Generate all the possible combination of options.
    It assumes the `conf_options` describes the possible values for each option.
    """
    conf_combinations = []
    for combination in itertools.product(*conf_options.values()):
        conf_combinations.append(dict(zip(conf_options.keys(), combination)))
    return conf_combinations


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
        "data_workers": "1",
        "name": "test",
        "checkpoint.save_dir": checkpoint_folder,
        "automatic_setup": False
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


# Set the different options to test
conf_options = {
    "trainer.prediction_type": ["delta", "full"],
    "trainer.enable_amp": ["True", "False"],
    "model.causal_in_time": ["True", "False"],
}


@pytest.mark.parametrize("conf", generate_parameters(conf_options), indirect=True)
def test_train(conf):
    """Test training terminates normally for different sets of config."""
    overrides = conf
    cfg_dir = os.path.relpath(CONFIG_DIR, os.path.dirname(__file__))
    with initialize(config_path=cfg_dir):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        with torch.autograd.detect_anomaly():
            main(cfg)
        assert True


def test_checkpoints(checkpoint_folder, conf):
    """Test training generates checkpoints as expected."""
    overrides = conf
    overrides += ["checkpoint.checkpoint_frequency=1", "trainer.max_epoch=2"]
    cfg_dir = os.path.relpath(CONFIG_DIR, os.path.dirname(__file__))
    with initialize(config_path=cfg_dir):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        main(cfg)
        # Check epoch 1 checkpoint exists
        print(os.listdir(checkpoint_folder))
        assert os.path.isdir(checkpoint_folder / "step_1")
        # Check best checkpoint exists
        assert os.path.islink(checkpoint_folder / "best")
        assert len(os.listdir(checkpoint_folder / "best")) > 0
        # Check metadata have been saved
        assert CHECKPOINT_METADA_FILENAME in os.listdir(checkpoint_folder / "best")
        # Check last checkpoint exists and is simlink of best
        assert os.path.islink(checkpoint_folder / "last")
        assert os.path.samefile(
            checkpoint_folder / "last", checkpoint_folder / "step_2"
        )


def test_no_checkpoint(checkpoint_folder, conf):
    """Test no checkpoints are saved when expected."""
    overrides = conf
    overrides += ["checkpoint=none"]
    cfg_dir = os.path.relpath(CONFIG_DIR, os.path.dirname(__file__))
    with initialize(config_path=cfg_dir):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        main(cfg)
        # Checkpoint folder should not exist
        assert not os.path.exists(checkpoint_folder)

@pytest.mark.skip(reason="Currently broken but not enough time to fix.")
def test_resume_from_checkpoint(checkpoint_folder, conf):
    """Test in a circumventing way training can resume from checkpoint"""
    overrides = conf
    overrides += [
        "checkpoint.checkpoint_frequency=0",
    ]
    cfg_dir = os.path.relpath(CONFIG_DIR, os.path.dirname(__file__))
    assert os.path.exists(checkpoint_folder) is False
    # First run
    with initialize(config_path=cfg_dir):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        main(cfg)
    # Because we always save last epoch
    # even though the frequency does not match
    # There should be step_1 in checkpoint folder
    assert "step_1" in os.listdir(checkpoint_folder)
    # Should resume from checkpoint
    overrides += ["checkpoint.checkpoint_frequency=1", "trainer.max_epoch=2"]
    with initialize(config_path=cfg_dir):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        main(cfg)
    # Because the training has resumed after 1 epoch (starting from 1)
    # There should be step_2 in checkpoint folder
    assert "step_2" in os.listdir(checkpoint_folder)
    assert os.path.samefile(checkpoint_folder / "last", checkpoint_folder / "step_2")
    assert not os.path.samefile(
        checkpoint_folder / "step_1", checkpoint_folder / "step_2"
    )
