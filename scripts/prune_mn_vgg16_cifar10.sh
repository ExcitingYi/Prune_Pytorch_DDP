#!/bin/bash

#- Job parameters

# (TODO)
# Please modify job name


#SBATCH -J test_ddp_vgg16_cifar10              # The job name
#SBATCH -o ./slurm_out_files/ret-%j.out        # Write the standard output to file named 'ret-<job_number>.out'
#SBATCH -e ./slurm_out_files/ret-%j.err        # Write the standard error to file named 'ret-<job_number>.err'


#- Resources

# (TODO)
# Please modify your requirements

#SBATCH -p r8nv-gpu-hw                  # Submit to 'nv-gpu' Partitiion
#SBATCH -t 0-12:00:00                # Run for a maximum time of 0 days, 12 hours, 00 mins, 00 secs
#SBATCH --nodes=1                    # Request N nodes
#SBATCH --gres=gpu:4                 # Request M GPU per node
#SBATCH --gres-flags=enforce-binding # CPU-GPU Affinity
#SBATCH --qos=gpu-short             # Request QOS Type

###
### The system will alloc 8 or 16 cores per gpu by default.
### If you need more or less, use following:
### #SBATCH --cpus-per-task=K            # Request K cores

### SBATCH --nodelist=gpu-a           # Request a specific list of hosts 
#SBATCH --constraint="Ampere&A30" # Request GPU Type: Volta(V100 or V100S) or RTX8000
###

#- Log information

echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Job run at:"
echo "$(hostnamectl)"

#- Load environments
source /tools/module_env.sh
# source /home/S/chenyi/.conda_init
echo $(module list)                       # list modules loaded
source /home/S/chenyi/.bashrc

##- Tools
module load cluster-tools/v1.0
module load slurm-tools/v1.0
module load cmake/3.15.7
module load git/2.17.1
module load vim/8.1.2424

##- language
# module load python3/3.6.8

##- CUDA
# module load cuda-cudnn/11.1-8.1.1

##- virtualenv
conda activate pt113cu116

echo $(module list)              # list modules loaded
echo $(which gcc)
echo $(which python)
echo $(which python3)

cluster-quota                    # nas quota

# nvidia-smi --format=csv --query-gpu=name,driver_version,power.limit # gpu info

nvidia-smi

#- Warning! Please not change your CUDA_VISIBLE_DEVICES
#- in `.bashrc`, `env.sh`, or your job script
echo "Use GPU ${CUDA_VISIBLE_DEVICES}"                              # which gpus
#- The CUDA_VISIBLE_DEVICES variable is assigned and specified by SLURM

#- Job step
python prune.py --data_root data --model vgg16 --dataset cifar10 \
    --batch_size 1024 --epochs 100 --lr 0.05 --wd 5e-5 --print_freq 10 \
    --lr_decay_milestones 40,70,90 --log_tag example --prune_type mn_prune \
    --weight_file ./checkpoints/scratch/cifar10_vgg16_scratch_ddp-example.pth --retrain_epoch 50 \
    --config_file ./prune_config/vgg16_cifar10_mn.yaml --prune_freq 5 \
    --ip 127.0.0.32 --port 23456

hostname

#- End
echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"
