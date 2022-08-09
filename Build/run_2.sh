#!/bin/bash


#SBATCH --mem=7G
#SBATCH --cpus-per-task=6

#SBATCH -w figuier


cd ..
source activate classif/bin/activate

##install requiremennts for the training
cd Build
pip install -r requirements.txt
cd ..


##Check missing requirements
#pip freeze > virtual_env_requirements.txt
python3 training_hb.py  \
			--data_file Embryon_RandomSplit\
                      	--criterion_name bce_balanced\
                        -bb ResNet18 -he Lstm\
                        --augmentation hflip vflip randombrightness fill_background\
                        --framestep 3\
                        --preload_cache\
                        -pb
exit 0
