#!/bin/bash

#SBATCH --mem=7G
#SBATCH --cpus-per-task=6

#SBATCH -w figuier

app="$(pwd)/../"
pythonEnv="${app}classif/"
. ${pythonEnv}"bin/activate"

##install requiremennts for the training
if [ ${VIRTUAL_ENV:(-7)} == "classif" ]; then 
        #pip install -r "$(pwd)/requirements.txt"
        cd ..
        python3 training.py  \
                                --data_file transferable_FV\
				--NBClass 2\
                                -bb ResNet18 -he ConvPooling\
                                --criterion_name bce\
                                --augmentation hflip vflip randombrightness fill_background\
                                --framestep 4\
				-pb
                                
else 
        echo "Virtual Environment issue, env name: ${$VIRTUAL_ENV}"
fi
##Check missing requirements
#pip freeze > virtual_env_requirements.txt
deactivate

exit 0
