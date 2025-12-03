from temporary_mppx_name.data.multidatamodule import MixedWellDataModule


def test_datamodule(dummy_dataset):
    well_base_path = dummy_dataset
    print(dummy_dataset)
    data_module = MixedWellDataModule(
        well_base_path=well_base_path,
        well_dataset_info={
            "dummy": {"include_filters": [], "exclude_filters": []},
        },
        batch_size=1,
        data_workers=1,
        max_samples=20,
    )
    assert hasattr(data_module, "train_dataset")
    assert hasattr(data_module, "train_dataloader")
    for batch_index, batch in enumerate(data_module.train_dataloader(), start=1):
        assert "input_fields" in batch
    assert batch_index == data_module.max_samples
