"""Directly inspired from https://pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html"""

from __future__ import annotations

import os
import os.path
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Callable, Optional, Tuple

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from controllable_patching_striding.optim.distributed_shampoo.distributed_shampoo import (
    DistributedShampoo,
)

CHECKPOINT_METADA_FILENAME = "metadata.pt"


def checkpoint_already_exists(checkpoint_dirname: str) -> bool:
    """Check if a given checkpoint already exists."""
    return os.path.exists(checkpoint_dirname)


def link_checkpoint(src_checkpoint, target_checkpoint):
    """Create a symbolic link to an already existing checkpoint.
    The link points to `src_checkpoint` and is named `target_checkpoint`.
    It allows avoiding expensive copies of checkpoints when they refer to the same data.
    To be used typically for saving last checkpoint that refers to an already existing one.
    """
    # Link already exists
    if os.path.exists(target_checkpoint) and os.path.islink(target_checkpoint):
        os.remove(target_checkpoint)
    os.symlink(src_checkpoint, target_checkpoint, target_is_directory=True)


def save_metadata(
    checkpoint_dir: str,
    epoch: Optional[int] = None,
    val_loss: Optional[float] = None,
    best_val_loss: Optional[float] = None,
):
    """Checkpoint information that do not require synchronization across GPUs,
    or which are already synchronized.
    To be used in combination of FSDP checkpointing strategy.

    """
    state_dict = {"epoch": epoch, "val_loss": val_loss, "best_val_loss": best_val_loss}
    torch.save(state_dict, os.path.join(checkpoint_dir, CHECKPOINT_METADA_FILENAME))


def on_future(fn: Callable, *args, **kwargs) -> Callable:
    """Wrap any callable to be run as future callback."""

    def future_wrapper(future: Future):
        return fn(*args, **kwargs)

    return future_wrapper


class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(
        self,
        model,
        optimizer=None,
    ):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        if isinstance(self.optimizer, DistributedShampoo):
            model_state_dict, _ = get_state_dict(self.model, [])
            optimizer_state_dict = self.optimizer.distributed_state_dict(
                key_to_param=self.model.named_parameters()
            )
        else:
            model_state_dict, optimizer_state_dict = get_state_dict(
                self.model, self.optimizer
            )
        return {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        if isinstance(self.optimizer, DistributedShampoo):
            set_state_dict(
                model=self.model,
                optimizers=[],
                model_state_dict=state_dict["model"],
                optim_state_dict=None,
                options=StateDictOptions(strict=False),
            )
            self.optimizer.load_distributed_state_dict(
                state_dict["optimizer"], key_to_param=self.model.named_parameters()
            )
        else:
            set_state_dict(
                model=self.model,
                optimizers=self.optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optimizer"],
                options=StateDictOptions(strict=False),
            )


class BaseCheckPointer(ABC):
    """Base class for checkpointing."""

    def __init__(self, save_dir: str, rank=0) -> None:
        self.save_dir = os.path.realpath(save_dir)
        self.rank = rank
        self._best_metrics: Optional[float] = None

    @abstractmethod
    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: str = None,  # Add checkpoint_path as an argument
    ):
        raise NotImplementedError

    @abstractmethod
    def save_if_necessary(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        val_loss: Optional[float] = None,
        epoch: Optional[int] = None,
        force: bool = False,
    ):
        raise NotImplementedError

    @property
    def last_checkpoint(self) -> str | None:
        """Return the real path of the last checkpoints in the directory."""
        last_checkpoint_dir = os.path.join(self.save_dir, "last")
        if os.path.exists(last_checkpoint_dir):
            return os.path.realpath(last_checkpoint_dir)
        else:
            warnings.warn("No last checkpoint found")
            return None

    @property
    def best_checkpoint(self) -> str | None:
        """Return the real path of the last checkpoints in the directory."""
        best_checkpoint_dir = os.path.join(self.save_dir, "best")
        if os.path.exists(best_checkpoint_dir):
            return os.path.realpath(best_checkpoint_dir)
        else:
            warnings.warn("No best checkpoint found")
            return None


class DummyCheckPointer(BaseCheckPointer):
    """Dummy class for checkpointing.
    It loads existing checkpoint to resume training.
    Does not actually checkpoint anything.
    """

    def load(
    self,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Optional[str] = None  # Add the checkpoint_path parameter
) -> Tuple[int | None, float | None]:
        # Use the provided checkpoint path directly
        if checkpoint_path:
            # Load the metadata directly from the provided checkpoint directory
            metadata_file = os.path.join(checkpoint_path, "metadata.pt")
            print(f"Loading metadata from {metadata_file}")
            checkpoint = torch.load(metadata_file, weights_only=False,)  # <-- Load the metadata file
        else:
            # If no checkpoint_path is provided, load from the "last" checkpoint
            last_checkpoint = self.last_checkpoint  # This is only if checkpoint_path is not provided
            assert last_checkpoint is not None and os.path.exists(last_checkpoint)
            metadata_file = os.path.join(self.save_dir, "last", CHECKPOINT_METADA_FILENAME)
            checkpoint = torch.load(metadata_file, weights_only=False,)  # Load from "last" checkpoint

        # Extract the epoch and validation loss from the checkpoint
        epoch = checkpoint.get("epoch", None)
        val_loss = checkpoint.get("val_loss", None)
        self._best_metrics = checkpoint.get("best_val_loss", None)

        # Load the model and optimizer state dicts
        state_dict = {"app": AppState(model, optimizer)}

        # Now pass the checkpoint directory to dcp.load, not the metadata file
        print('ckpt path before dcp', checkpoint_path)
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=checkpoint_path,# or last_checkpoint,  # Use the correct directory path
        )

        """# Check if the weights are loaded
        loaded_state_dict = app_state.state_dict()["model"]  # Get the loaded state dict
        current_state_dict = model.state_dict()  # Get the current model state dict

        val1 = loaded_state_dict['space_bag.linear.weight']
        val2 = current_state_dict['space_bag.linear.weight']

        print("Loaded state dict keys:", loaded_state_dict.keys())
        print("Current state dict keys:", current_state_dict.keys())
        if torch.equal(val1, val2):
            print("Weights match!")
        else:
            print("Weights do not match!")"""


        return epoch, val_loss


class CheckPointer(DummyCheckPointer):
    """Class to checkpoint training state_dict under FSDP strategy."""

    def __init__(
        self,
        save_dir: str,
        save_best: bool = True,
        checkpoint_frequency: int = 0,
        rank: int = 0,
    ):
        super().__init__(save_dir, rank)
        print("Checkpointing to", save_dir)
        if rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        self.save_best = save_best
        self.checkpoint_frequency = checkpoint_frequency

    def save_if_necessary(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        val_loss: Optional[float] = None,
        epoch: Optional[int] = None,
        force: bool = False,
    ) -> Optional[Future]:
        """Check if checkpoints must be saved.
        If so triggers the asynchronous writing of the file.
        Upon writing link additional checkpoints to the written file.
        Returns an optional future.

        Args
        ----
        model: Model whose state is to save
        optimizer: Optimizer state to save
        val_loss: Loss saved as checkpoint metadata and used to save best model
        epoch: Epoch saved as checkpoint metadata and used to save the model every n epochs
        force: Boolean flag to force saving the model, preempts saving every n epoch behavior
        """
        state_dict = {"app": AppState(model, optimizer)}
        checkpoint_future = None
        checkpoint_dirnames = []
        # Save checkpoint based on epoch
        # Those checkpoints as they are absolute (comparatively to best) should be the ones saved
        # Other checkpoints can refer to these ones and thus be linked to avoid hard copies
        save_this_epoch = False
        epoch_checkpoint_dirname = os.path.join(self.save_dir, f"step_{epoch}")
        # Force saving checkpoint which does not already exists
        # Typically occurs when saving checkpoint at the end of training
        if checkpoint_already_exists(epoch_checkpoint_dirname):
            save_this_epoch = False
        elif force:
            save_this_epoch = True
        # Epoch number triggers checkpointing
        elif (
            epoch is not None
            and self.checkpoint_frequency
            and (epoch % self.checkpoint_frequency) == 0
        ):
            save_this_epoch = True
        if save_this_epoch:
            checkpoint_dirnames.append(epoch_checkpoint_dirname)

        # Best metrics triggers checkpointing
        if self.save_best:
            assert (
                val_loss is not None
            ), "Expect to save best metrics but no metrics provided."
            if self._best_metrics is None or val_loss < self._best_metrics:
                self._best_metrics = val_loss
                checkpoint_dirnames.append(os.path.join(self.save_dir, "best"))

        # Several files should be saved for the same checkpoints
        if checkpoint_dirnames:
            checkpoint_dirnames.append(os.path.join(self.save_dir, "last"))
            actual_checkpoint_dirname = checkpoint_dirnames[0]
            # Only save the first checkpoints
            checkpoint_future = dcp.async_save(
                state_dict, checkpoint_id=actual_checkpoint_dirname
            )
            # Save already synchronized data on rank 0 only
            # To be used typically for saving epoch and loss
            if self.rank == 0:
                checkpoint_future.add_done_callback(
                    on_future(
                        save_metadata,
                        checkpoint_dir=actual_checkpoint_dirname,
                        epoch=epoch,
                        val_loss=val_loss,
                        best_val_loss=self._best_metrics,
                    )
                )
            # Link the other checkpoints to the one that has been saved
            # Only to be performed once, hence on rank 0
            if self.rank == 0:
                for linked_checkpoint_filename in checkpoint_dirnames[1:]:
                    checkpoint_future.add_done_callback(
                        on_future(
                            link_checkpoint,
                            actual_checkpoint_dirname,
                            linked_checkpoint_filename,
                        )
                    )

        if force and checkpoint_future is not None:
            # Make checkpoint saving synchronous
            checkpoint_future.result()
            checkpoint_future = None

        return checkpoint_future
