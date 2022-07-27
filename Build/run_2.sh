#!/bin/bash

#SBATCH -o output.txt

#SBATCH --mem=7G
#SBATCH --cpus-per-task=6

#SBATCH -w erable

cd ..
source activate classif/bin/activate

##install requiremennts for the training
#pip install -r requirements.txt


##Check missing requirements
#pip freeze > virtual_env_requirements.txt
python3 training.py  \
			--data_file Embryon_RandomSplit\
                      	--criterion_name bce_balanced\
                        -bb ResNet18 -he Gru\
                        --augmentation hflip\
                        --framestep 1\
                        --preload_cache\
                        -pb
exit 0
