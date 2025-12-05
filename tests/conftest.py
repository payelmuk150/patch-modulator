import pathlib
import shutil

import pytest
from the_well.utils.dummy_data import write_dummy_data


@pytest.fixture()
def dummy_dataset(tmp_path):
    well_data_folder = tmp_path / "well_data"
    well_data_folder.mkdir()
    for split in ["train", "valid", "test"]:
        split_dir = well_data_folder / "dummy" / "data" / split
        split_dir.mkdir(parents=True)
        write_dummy_data(split_dir / "data.hdf5")
    return well_data_folder


@pytest.fixture()
def checkpoint_folder(tmp_path: pathlib.Path):
    """Create and clean a temporary folder for checkpoints"""
    yield tmp_path / "checkpoints"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
