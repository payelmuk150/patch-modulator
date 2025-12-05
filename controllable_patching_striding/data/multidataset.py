import logging
import os
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
import torch
from the_well.data import WellDataset
from the_well.data.utils import WELL_DATASETS, flatten_field_names
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MixedWellDataset(Dataset):
    """
    Combination of multiple Well datasets. Returns data in B x T x H [x W [x D]] x C format.

    Train/Test/Valid is assumed to occur on a folder level and this is not performed in this
    object.

    Most parameters are passed to inner datasets.

    Parameters
    ----------
    paths :
        Path to directory of HDF5 files, one of path or well_base_path+well_dataset_name
          must be specified
    normalization_path:
        Path to normalization constants - assumed to be in same format as constructed data.
    well_base_path :
        Path to well dataset directory, only used with dataset_name
    well_dataset_info:
        Dictionary containing for each dataset:
        - include_filters: List of strings to filter files to include
        - exclude_filters: List of strings to filter files to exclude
        - path: Optional custom path for this specific dataset
    well_split_name :
        Name of split to load - options are 'train', 'valid', 'test'
    include_filters :
        Only include files whose name contains at least one of these strings
    exclude_filters :
        Exclude any files whose name contains at least one of these strings
    use_normalization:
        Whether to normalize data in the dataset
    include_normalization_in_sample: bool, default=False
        Whether to include normalization constants in the sample
    n_steps_input :
        Number of steps to include in each sample
    n_steps_output :
        Number of steps to include in y
    dt_stride :
        Minimum stride between samples
    max_dt_stride :
        Maximum stride between samples
    flatten_tensors :
        Whether to flatten tensor valued field into channels
    cache_small :
        Whether to cache all values that do not vary in time or sample
          in memory for faster access
    max_cache_size :
        Maximum numel of constant tensor to cache
    return_grid :
        Whether to return grid coordinates
    boundary_return_type : options=['padding', 'mask', 'exact']
        How to return boundary conditions. Currently only padding supported.
    full_trajectory_mode :
        Overrides to return full trajectory starting from t0 instead of samples
            for long run validation.
    name_override :
        Override name of dataset (used for more precise logging)
    transforms :
        Dict of transforms to apply to data. Each key should be a dataset name.
    """

    def __init__(
        self,
        *,
        well_base_path: str,
        well_dataset_info: Dict[
            str,
            Dict[
                Literal["include_filters", "exclude_filters", "path"], List[str] | str
            ],
        ],
        path: Optional[str] = None,
        tie_fields: bool = True,
        use_effective_batch_size: bool = False,
        prefetch_field_names: bool = True,
        normalization_path: Optional[str] = "../stats/",
        well_split_name: str = "train",
        use_normalization: bool = False,
        max_rollout_steps=100,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        min_dt_stride: int = 1,
        max_dt_stride: int = 1,
        flatten_tensors: bool = True,
        cache_small: bool = True,
        max_cache_size: float = 1e9,
        return_grid: bool = True,
        boundary_return_type: str = "padding",
        full_trajectory_mode: bool = False,
        name_override: Optional[str] = None,
        transform: Optional[Dict[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
    ):
        super().__init__()
        # Global dicts used by Mixed DSET.
        self.well_base_path = well_base_path
        self.well_datasets = well_dataset_info
        self.prefetch_field_names = prefetch_field_names
        self.tie_fields = tie_fields
        self.well_split_name = well_split_name
        self.sub_dsets = []
        self.use_effective_batch_size = use_effective_batch_size
        self.effective_batch_sizes = []
        self.offsets = [0]
        self.dset_to_metadata: dict[str, Any] = {}
        self.well_dataset_info = well_dataset_info

        if transform is not None:
            assert set(list(transform.keys())).issubset(set(well_dataset_info.keys()))

        for dataset_name, info in well_dataset_info.items():
            include_filters = info.get("include_filters", [])
            exclude_filters = info.get("exclude_filters", [])
            dataset_path = info.get("path", None)
            subdset = WellDataset(
                path=dataset_path,
                normalization_path=normalization_path,
                well_base_path=well_base_path,
                well_dataset_name=dataset_name,
                well_split_name=well_split_name,
                include_filters=include_filters,
                exclude_filters=exclude_filters,
                use_normalization=use_normalization,
                max_rollout_steps=max_rollout_steps,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                min_dt_stride=min_dt_stride,
                max_dt_stride=max_dt_stride,
                flatten_tensors=flatten_tensors,
                cache_small=cache_small,
                max_cache_size=max_cache_size,
                return_grid=return_grid,
                boundary_return_type=boundary_return_type,
                full_trajectory_mode=full_trajectory_mode,
                name_override=name_override,
                transform=(
                    transform[dataset_name]
                    if transform is not None and dataset_name in transform
                    else None
                ),
            )
            try:
                offset = len(subdset)
                self.offsets.append(self.offsets[-1] + offset)
            except ValueError:
                raise ValueError(
                    f"Dataset {path} is empty. Check that n_steps < trajectory_length in file."
                )
            self.sub_dsets.append(subdset)
            self.dset_to_metadata[dataset_name] = subdset.metadata
            if self.use_effective_batch_size:
                raise NotImplementedError  # TODO implement effective batch size logic
            else:
                effective_batch_size = 1
            self.effective_batch_sizes.append(effective_batch_size)
            self.offsets[0] = -1  # So 0 is in the first segment

        self.field_to_index_map = self._build_subset_dict()

    
    def _build_subset_dict(self) -> Dict[str, int]:
        # Maps fields to subsets of variables
        field_to_index = {}
        max_index = 0
        if self.prefetch_field_names:
            for dataset_name in WELL_DATASETS:
                try:
                    temp_dset = WellDataset(
                        well_base_path=self.well_base_path,
                        well_dataset_name=dataset_name,
                        well_split_name=self.well_split_name,
                        use_normalization=False,  # Don't need normalization to get this data
                    )
                except Exception:
                    logger.warning(f"Failed to load {dataset_name} dataset")
                    continue
                metadata = temp_dset.metadata
                field_names = flatten_field_names(metadata)
                for field_name in field_names:
                    # If we're not tying field names, then add dataset name to field name for the key
                    if not self.tie_fields:
                        field_name = f"{dataset_name}_{field_name}"
                    if field_name not in field_to_index:
                        field_to_index[field_name] = max_index
                        max_index += 1
            # If we added any extras, make sure they're represented as well
            for dataset_name, info in self.well_dataset_info.items():
                if dataset_name in WELL_DATASETS and self.prefetch_field_names:
                    continue  # Already processed this dataset in the previous loop
                dataset_path = info.get("path", None)
                if dataset_path is not None:
                    temp_dset = WellDataset(
                        path=dataset_path,
                        well_split_name=self.well_split_name,
                        use_normalization=False,
                    )
                elif dataset_name in WELL_DATASETS:
                    temp_dset = self.inner_dataset_type(
                        well_base_path=self.well_base_path,
                        well_dataset_name=dataset_name,
                        well_split_name=self.well_split_name,
                        use_normalization=False,  # Don't need normalization to get this data
                    )
                else:
                    raise ValueError(
                        f"Unknown dataset {dataset_name}. Please provide path."
                    )
                metadata = temp_dset.metadata
                field_names = flatten_field_names(metadata)
                for field_name in field_names:
                    # If we're not tying field names, then add dataset name to field name for the key
                    if not self.tie_fields:
                        field_name = f"{dataset_name}_{field_name}"
                    if field_name not in field_to_index:
                        field_to_index[field_name] = max_index
                        max_index += 1
        return field_to_index

    def __getitem__(self, index):
        file_idx = (
            np.searchsorted(self.offsets, index, side="right") - 1
        )  # which dataset are we are on
        local_idx = index - max(self.offsets[file_idx], 0)
        try:
            data = self.sub_dsets[file_idx][local_idx]
        except Exception:
            raise IndexError(
                "FAILED AT ", file_idx, local_idx, index, int(os.environ.get("RANK", 0))
            )
        current_metadata = self.sub_dsets[file_idx].metadata
        field_names = flatten_field_names(current_metadata)
        if not self.tie_fields:
            field_names = [
                f"{current_metadata.dataset_name}_{field}" for field in field_names
            ]
        field_indices = [self.field_to_index_map[field] for field in field_names]
        data["field_indices"] = torch.tensor(field_indices)
        data["metadata"] = current_metadata
        return data

    def __len__(self):
        return sum([len(dset) for dset in self.sub_dsets])


if __name__ == "__main__":
    well_base_path = "/mnt/home/polymathic/ceph/the_well/"
    data = MixedWellDataset(
        well_base_path=well_base_path,
        well_dataset_info={
            "active_matter": {"include_filters": [], "exclude_filters": []},
            "planetswe": {"include_filters": [], "exclude_filters": []},
        },
    )

    for i in range(len(data)):
        x = data[i]
        if i % 1000 == 0:
            print(x)

    # print(len(data))
    # print(data[0])