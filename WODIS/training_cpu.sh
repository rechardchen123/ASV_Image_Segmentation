#!/bin/bash -l
# Batch script to run a GPU job on Myriad under SGE.

# 0. Force bash as the executing shell.
#$ -S /bin/bash

#2. Request half hour of wallclock time (format hours:minutes:second).
#$ -l h_rt=10:00:0

#3. Request 8 gigabyte of RAM (must be an integer)
#$ -l mem=16G

#4. Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=15G

# Request 16 cores.
#$ -pe smp 16

#5. Set the name of the job.
#$ -N ASV_WODIS

#6. Set the working directory to somewhere in your scratch space.  This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/ucesxc0/Scratch/output/asv_image_segmentation/WODIS/



#8. activate the virtualenv
source activate pytorch
#9.  load the cuda module
module unload compilers
module load compilers/gnu/4.9.2
#module load cuda/10.1.243/gnu-4.9.2
#module load cudnn/7.6.5.32/cuda-10.1
#10. Run job
./train.py

source deactivate





