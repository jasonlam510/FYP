## Using miniconda Python in batch job
#!/bin/bash -l
# The -l above is required to get the full environment with modules
# Set the allocation to be charged for this job
# not required if you have set a default allocation
#SBATCH -A edu25.DD2356
# The name of the script is myjob
#SBATCH -J jasons' fyp
# Only 6 hour wall-clock time will be given to this job
#SBATCH -t 6:00:00
# Number of tasks
#SBATCH -n 1
# Job partition
#SBATCH -p shared
# load the Miniconda module
ml PDC/23.12
ml miniconda3/24.7.1-0-cpeGNU-23.12
# if you need the custom conda environment:
conda activate my-conda-env
# execute the program
srun -n 1 python main.py
# to deactivate the Miniconda environment
conda deactivate
