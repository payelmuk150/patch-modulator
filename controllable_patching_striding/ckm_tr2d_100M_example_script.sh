#!/bin/bash
#SBATCH --time=21:00:00
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH -J CSM-TRL2D
#SBATCH --output=test_tr2d_release_ckm.log

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
# export NCCL_DEBUG=TRACE

# Set CUDA_LAUNCH_BLOCKING to help debug the CUDA errors
export CUDA_LAUNCH_BLOCKING=1

# module load python cuda cudnn gcc hdf5
# Activate the virtual environment with all the dependencies
source /mnt/home/pmukhopadhyay/projects/multiple_physics_pretraining/myenv3.10/bin/activate # Replace with your own environment
export PYTHONPATH=$PYTHONPATH:/mnt/home/pmukhopadhyay/projects/patch-modulator/controllable_patching_striding/ # Replace with your own path

# Launch the training script
srun python `which torchrun` \
	--nnodes=$SLURM_JOB_NUM_NODES\
	--nproc_per_node=$SLURM_GPUS_PER_NODE \
	--rdzv_id=$SLURM_JOB_ID \
		--rdzv_backend=c10d \
		--rdzv_endpoint=$SLURMD_NODENAME:29500 \
	controllable_patching_striding/train.py distribution=fsdp server=rusty optimizer.lr=0.0001 logger.wandb_project_name="FLEXIBLE_PATCHING_EXPERIMENTS" \
			data.module_parameters.batch_size=2 data.module_parameters.max_samples=100 model.hidden_dim=768 model.groups=12 model.processor_blocks=12 model.drop_path=.1 \
			model/processor/space_mixing=full_spatial_attention model.processor.space_mixing.num_heads=12 model.processor.time_mixing.num_heads=12 \
			model.causal_in_time=True model.jitter_patches=False \
			model/encoder=flexivit_encoder \
			model/decoder=flexivit_decoder \
            model.encoder.variable_deterministic_ds=False\
            model.encoder.base_kernel_size2d="[[4,4],[4,4]]"\
            model.encoder.kernel_scales_seq="[[2,2], [4,2], [4,4]]"\
            model.decoder.base_kernel_size2d="[[4,4], [4,4]]" \
			model.infer="[4,4]"\
			model.twod_only=True\
            model.threed_only=False\
			trainer.prediction_type=delta trainer.max_rollout_steps=10 trainer.infer_type=fixed data=TRL_2D trainer.max_epoch=601 data_workers=4 auto_resume=False
