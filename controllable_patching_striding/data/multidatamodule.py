import logging
from typing import Dict, List, Literal

from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Sampler
from torch.utils.data._utils.collate import default_collate

from .mixed_dset_sampler import MultisetSampler
from .multidataset import MixedWellDataset

logger = logging.getLogger(__name__)


def metadata_aware_collate(batch):
    """Collate function that is aware of the metadata of the dataset."""
    # Metadata constant per batch
    metadata = batch[0]["metadata"]
    # Remove metadata from current dicts
    [sample.pop("metadata") for sample in batch]
    batch = default_collate(batch)  # Returns stacked dictionary
    batch["metadata"] = metadata
    return batch


class MixedWellDataModule:
    def __init__(
        self,
        well_base_path: str,
        well_dataset_info: Dict[
            str,
            Dict[
                Literal["include_filters", "exclude_filters", "path"], List[str] | str
            ],
        ],
        batch_size: int,
        max_rollout_steps: int = 100,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        min_dt_stride: int = 1,
        max_dt_stride: int = 1,
        world_size: int = 1,
        rank: int = 1,
        data_workers: int = 4,
        epoch: int = 0,
        max_samples: int = 2000,
    ):
        """Data module class to yield batches of samples.

        Parameters
        ----------
        well_base_path:
            Path to the base directory for the Well dataset.
        well_dataset_info:
            Dictionary containing for each dataset:
            - include_filters: List of strings to filter files to include
            - exclude_filters: List of strings to filter files to exclude
            - path: Optional custom path for this specific dataset
        batch_size:
            Size of the batches yielded by the dataloaders
        max_rollout_steps:
            Maximum number of steps to rollout for the full trajectory mode.
        n_steps_input:
            Number of simulation time frames to include in input.
        n_steps_output:
            Number of simulation time frames to include in output.
        dt_stride:
            Stride for the time dimension.
        world_size:
            Number of total processes in the distributed setting.
        rank:
            Rank of the current GPU in the full torchrun world.
        data_workers:
            Number of workers for the dataloaders in the given process.
        epoch:
            Current epoch number.
        max_samples:
            Maximum number of samples to use for a single training loop.
        """
        # Train is a single mixed dataset
        self.train_dataset = MixedWellDataset(
            well_base_path=well_base_path,
            well_dataset_info=well_dataset_info,
            well_split_name="train",
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            min_dt_stride=min_dt_stride,
            max_dt_stride=max_dt_stride
        )
        # In Val/Test, we want stats for each dataset
        # but we still use MixedWellDataset to handle the extra info (field indices, etc.)
        self.val_datasets = [
            MixedWellDataset(
                well_base_path=well_base_path,
                well_dataset_info={key: well_dataset_info[key]},
                well_split_name="valid",
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                min_dt_stride=min_dt_stride,
                max_dt_stride=min_dt_stride
            )
            for key in well_dataset_info
        ]

        self.rollout_val_datasets = [
            MixedWellDataset(
                well_base_path=well_base_path,
                well_dataset_info={key: well_dataset_info[key]},
                well_split_name="valid",
                max_rollout_steps=max_rollout_steps,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                full_trajectory_mode=True,
                min_dt_stride=min_dt_stride,
                max_dt_stride=min_dt_stride
            )
            for key in well_dataset_info
        ]

        self.test_datasets = [
            MixedWellDataset(
                well_base_path=well_base_path,
                well_dataset_info={key: well_dataset_info[key]},
                well_split_name="test",
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                min_dt_stride=min_dt_stride,
                max_dt_stride=min_dt_stride
            )
            for key in well_dataset_info
        ]

        self.rollout_test_datasets = [
            MixedWellDataset(
                well_base_path=well_base_path,
                well_dataset_info={key: well_dataset_info[key]},
                well_split_name="test",
                max_rollout_steps=max_rollout_steps,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                full_trajectory_mode=True,
                min_dt_stride=min_dt_stride,
                max_dt_stride=min_dt_stride
            )
            for key in well_dataset_info
        ]
        self.batch_size = batch_size
        self.world_size = world_size
        self.data_workers = data_workers
        self.rank = rank
        self.epoch = epoch
        self.max_samples = max_samples

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def train_dataloader(self) -> DataLoader:
        if self.is_distributed:
            base_sampler: type[Sampler] = DistributedSampler
        else:
            base_sampler = RandomSampler

        sampler = MultisetSampler(
            self.train_dataset,
            base_sampler,
            self.batch_size,  # seed=seed,
            distributed=self.is_distributed,
            max_samples=self.max_samples,  # TODO Fix max_samples later
            rank=self.rank,
        )
        shuffle = sampler is None

        return DataLoader(
            self.train_dataset,
            num_workers=self.data_workers,
            pin_memory=True,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            sampler=sampler,
            collate_fn=metadata_aware_collate,
        )

    def build_loaders_from_dset_list(self, dset_list, batch_size=1) -> List[DataLoader]:
        dataloaders = []
        for dataset in dset_list:
            # If distributed, don't replicate across GPUs
            if self.is_distributed:
                # However, for large enough worlds, we need drop_last=False which causes some replication
                sampler: Sampler = DistributedSampler(dataset, seed=0, drop_last=False)
            else:
                sampler = RandomSampler(
                    dataset
                )  # TODO - Add seed here to make this consistent.

            dataloaders.append(
                DataLoader(
                    dataset,
                    num_workers=self.data_workers,
                    pin_memory=True,
                    batch_size=batch_size,
                    shuffle=None,  # Sampler is set
                    drop_last=True,
                    sampler=sampler,
                    #sampler=None, # This is only for rollout evals
                    collate_fn=metadata_aware_collate,
                )
            )
        return dataloaders

    def val_dataloaders(self) -> List[DataLoader]:
        return self.build_loaders_from_dset_list(self.val_datasets, self.batch_size)

    def rollout_val_dataloaders(self) -> List[DataLoader]:
        return self.build_loaders_from_dset_list(self.rollout_val_datasets, 1)

    def test_dataloaders(self) -> List[DataLoader]:
        return self.build_loaders_from_dset_list(self.test_datasets, self.batch_size)

    def rollout_test_dataloaders(self) -> List[DataLoader]:
        return self.build_loaders_from_dset_list(self.rollout_test_datasets, 1)


if __name__ == "__main__":
    well_base_path = "/mnt/home/polymathic/ceph/the_well/"
    data = MixedWellDataModule(
        well_base_path=well_base_path,
        well_dataset_info={
            "active_matter": {"include_filters": [], "exclude_filters": []},
            "planetswe": {"include_filters": [], "exclude_filters": []},
        },
        batch_size=32,
        data_workers=4,
    )

    for x in data.train_dataloader():
        print(x)
        break
