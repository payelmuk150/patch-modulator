import pytest
from the_well.utils.dummy_data import write_dummy_data


@pytest.fixture()
def dummy_dataset(tmp_path):
    well_data_folder = tmp_path / "well_data"
    well_data_folder.mkdir()
    for split in ["train", "valid", "test"]:
        split_dir = well_data_folder / "datasets" / "dummy_placeholder" / "data" / split
        split_dir.mkdir(parents=True)
        write_dummy_data(split_dir / "data.hdf5")
    return well_data_folder
