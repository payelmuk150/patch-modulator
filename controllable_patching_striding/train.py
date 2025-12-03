import functools
import logging
import os
import os.path as osp
from typing import Dict, Optional, cast

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary

# TODO - rewrite this for torchrun
from controllable_patching_striding.data import MixedWellDataModule
from controllable_patching_striding.data.well_to_multi_transformer import (
    ChannelsFirstWithTimeFormatter,
)
from controllable_patching_striding.optim.distributed_shampoo.shampoo_types import (
    FSDPShampooConfig,
    HSDPShampooConfig,
)
from controllable_patching_striding.optim.distributed_shampoo.utils.shampoo_fsdp_utils import (
    compile_fsdp_parameter_metadata,
)
from controllable_patching_striding.trainer.checkpoints import BaseCheckPointer
from controllable_patching_striding.trainer.training import Trainer
from controllable_patching_striding.utils.distribution_utils import (
    configure_distribution,
    distribute_model,
)
from controllable_patching_striding.utils.experiment_utils import configure_experiment

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

# Retrieve configuration for hydra
CONFIG_DIR = osp.join(osp.dirname(__file__), "configs")
CONFIG_NAME = "config"
CONFIG_PATH = osp.join(CONFIG_DIR, f"{CONFIG_NAME}.yaml")
assert osp.isfile(CONFIG_PATH), f"Configuration {CONFIG_PATH} is not an existing file."
logger.info(f"Run training script for {CONFIG_PATH}")


def train(
    cfg: DictConfig,
    experiment_name: str,
    experiment_folder: str,
    viz_folder: str,
    is_distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    local_rank: int = 0,
    mesh: Optional[torch.distributed.device_mesh.DeviceMesh] = None,
):
    """Instantiate the different objects required for training and run the training loop."""

    logger.info(f"Instantiate datamodule {cfg.data.wandb_data_name}")
    datamodule: MixedWellDataModule = instantiate(
        cfg.data.module_parameters,
        world_size=world_size,
        rank=rank,
        data_workers=cfg.data_workers,
        well_base_path=cfg.data.well_base_path,
    )
    # TODO - currently enforcing MPP format, but should allow for other types
    # Retrieve the number of fields used in training
    # from the mapping of field to index
    total_input_fields = max(datamodule.train_dataset.field_to_index_map.values()) + 1

    logger.info(
        f"Instantiate model {cfg.model._target_}",
    )
    model: torch.nn.Module = instantiate(
        cfg.model,
        n_states=total_input_fields,
    )
    if rank == 0:
        summary(model, depth=5)

    logger.info(
        f"Assigning distribution strategy: {cfg.distribution.distribution_type}"
    )
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(local_rank)}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model = distribute_model(model, cfg, mesh)

    logger.info(f"Instantiate optimizer {cfg.optimizer._target_}")
    _partial = False
    if "DistributedShampoo" in cfg.optimizer._target_:
        # See doc at https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/README.md
        # For DDP, we use the default cfg.optimizer.distributed_config configuration. Otherwise, we override it.
        distribution_type = cfg.distribution.distribution_type.upper()
        if distribution_type == "LOCAL":
            cfg.optimizer.distributed_config = (
                None  # local distribution does not require any special configuration
            )
        elif distribution_type == "DDP":
            pass
        elif distribution_type == "FSDP":
            distributed_config = FSDPShampooConfig(
                param_to_metadata=compile_fsdp_parameter_metadata(model)
            )
            _partial = True  # Hack due to Hydra limitations
        elif distribution_type == "HSDP":
            logger.warning(
                "HSDP requires torch>2.4.1 (_MeshEnv._get_all_submeshes is not implemented in <=2.4.1). Waiting for a stable release of torch before updating requirements."
            )
            distributed_config = HSDPShampooConfig(
                param_to_metadata=compile_fsdp_parameter_metadata(model),
                device_mesh=mesh,
                num_trainers_per_group=cfg.optimizer.distributed_config.num_trainers_per_group,
            )
            _partial = True  # Hack due to Hydra limitations
        else:
            raise ValueError(f"Unknown distribution type {distribution_type}")

    optimizer: torch.optim.Optimizer | functools.partial = instantiate(
        cfg.optimizer, params=model.parameters(), _partial_=_partial
    )
    if _partial:  # Only for distributed_shampoo
        # Just a hack to instantiate the optimizer with the correct distributed_config parameter
        # (Hydra forces us to do it this way)
        optimizer = optimizer(distributed_config=distributed_config)

    # Set start epoch to 0 before potential retrieval from checkpoint
    start_epoch = 1
    val_loss = torch.tensor(float("inf"))
    logger.info(f"Instantiate checkpointer {cfg.checkpoint._target_}")
    checkpointer: BaseCheckPointer = instantiate(cfg.checkpoint)
    
    # ------------------------------------------------------------------------------
    # Checkpoint loading
    #
    # If no checkpoint path is provided, training starts from scratch.
    #
    # To run validation from a saved checkpoint, set
    # `custom_checkpoint_path` to the *directory containing the checkpoint file*.
    #
    # In this codebase, checkpoints follow a directory structure such as:
    #
    #   .../infer-<experiment_name>/<run_id>/checkpoints/step_<N>/
    #
    # Therefore, the value of `custom_checkpoint_path` should be the full path to
    # the `step_<N>/` directory (not the parent folder, and not the file inside it).
    #
    # Example:
    #   /mnt/home/.../infer-default_name-well2-delta-Isotr[VstriLearnedPad-True-...]/2/checkpoints/step_x/
    #
    # Set `custom_checkpoint_path` below to the one you want to load.
    # ------------------------------------------------------------------------------
    # TODO: MOVE THIS TO A CONFIG FILE.
    custom_checkpoint_path = ""   

    if os.path.exists(custom_checkpoint_path):
        logger.info(f"Resuming from checkpoint: {custom_checkpoint_path}")
        # Load the model weights and optimizer state if needed
        epoch, val_loss = checkpointer.load(model, optimizer, checkpoint_path=custom_checkpoint_path)

        # Ensure initial_lr is set for each parameter group
        for param_group in optimizer.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']

        logger.info(f"Infer at epoch {epoch} with validation loss {val_loss}")
        start_epoch = 1 if epoch is None else epoch + 1
        last_epoch = epoch - 1  # Set last_epoch to the last completed epoch
    else:
        logger.info(f"Checkpoint path {custom_checkpoint_path} does not exist. Starting from scratch.")
        start_epoch = 1  # Start from the first epoch if no checkpoint
        last_epoch = -1  # No previous epoch

    if hasattr(cfg, "lr_scheduler"):

        # Instantiate LR scheduler
        logger.info(f"Instantiate learning rate scheduler {cfg.lr_scheduler._target_}")
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = instantiate(
            cfg.lr_scheduler,
            optimizer=optimizer,
            max_epochs=cfg.trainer.max_epoch,
            warmup_start_lr=cfg.optimizer.lr * 0.1,
            eta_min=cfg.optimizer.lr * 0.1,
            last_epoch=last_epoch  # Use the last_epoch from the checkpoint
        )
    else:
        logger.info("No learning rate scheduler")
        lr_scheduler = None

    if rank == 0:
        logger.info(f"Final configuration:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Instantiate trainer {cfg.trainer._target_}")
    #val1 = model.state_dict()['space_bag.linear.weight']
    trainer: Trainer = instantiate(
        cfg.trainer,
        experiment_name=experiment_name,
        viz_folder=viz_folder,
        model=model,
        datamodule=datamodule,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpointer=checkpointer,
        device=device,
        is_distributed=is_distributed,
        distribution_type=cfg.distribution.distribution_type,
        rank=rank,
        formatter=ChannelsFirstWithTimeFormatter,  # TODO change this to function of model
        wandb_logging=cfg.logger.wandb,
        start_epoch=start_epoch,
        start_val_loss=val_loss,
    )

    if cfg.validation_mode:
        trainer.validate()
    else:
        # Save config to directory folder
        if rank == 0:
            with open(osp.join(experiment_folder, "extended_config.yaml"), "w") as f:
                OmegaConf.save(cfg, f)
        trainer.train()


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    # Torch optimization settings
    torch.set_float32_matmul_precision("high")  # Use TF32 when supported
    # Retrieve multiple processes context to setup DDP
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = (
        cfg.distribution.distribution_type.upper() != "LOCAL" and world_size > 1
    )

    # Since configure_experiment uses distributed logic, distribution must be set up first
    mesh = configure_distribution(cfg)
    (
        cfg,
        experiment_name,
        experiment_folder,
        checkpoint_folder,
        artifact_folder,
        viz_folder,
    ) = configure_experiment(cfg, rank, is_distributed)

    logger.info(f"Run experiment {experiment_name}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    # Initiate wandb logging

    # Make sure we're logging the true batch size
    config_for_wandb = cast(Dict, OmegaConf.to_container(cfg, resolve=True))
    config_for_wandb["world_size"] = world_size
    # Global batch size is microbatch size * number of GPUs * gradient accumulation steps
    config_for_wandb["global_batch_size"] = (
        cfg.data.module_parameters.batch_size * world_size
    ) * cfg.trainer.grad_acc_steps
    if rank == 0 and cfg.logger.wandb:
        wandb.init(
            project=cfg.logger.wandb_project_name,
            group=f"{cfg.data.wandb_data_name}",
            config=config_for_wandb,
            name=experiment_name,
        )
    train(
        cfg,
        experiment_name,
        experiment_folder,
        viz_folder,
        is_distributed,
        world_size,
        rank,
        local_rank,
        mesh=mesh,
    )
    if rank == 0 and cfg.logger.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
