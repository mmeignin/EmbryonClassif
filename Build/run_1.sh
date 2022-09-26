#!/bin/bash

#SBATCH --mem=50G
#SBATCH --cpus-per-task=10

#SBATCH -w figuier

app="$(pwd)/../"
pythonEnv="${app}classif/"
. ${pythonEnv}"bin/activate"

##install requiremennts for the training
if [ ${VIRTUAL_ENV:(-7)} == "classif" ]; then 
        cd ..
        python3 training.py  \
                                --data_file Binary_FV\
				--NBClass 2\
                                -bb ResNet18 -he Lstm\
                                --criterion_name bce_balanced\
                                --augmentation hflip vflip randombrightness fill_background\
                                --framestep 4\
				--preload_cache
        echo "Training is over !"
	
else 
        echo "Environment Name should be classif: ${$VIRTUAL_ENV}"
        echo "Creating Environment and installing dependency"
        python3 -m venv pythonEnv
        pip install -r "$(pwd)/requirements.txt"
        echo "Environment Name should now be classif: ${$VIRTUAL_ENV}"
        ##Check missing requirements
        #pip freeze > virtual_env_requirements.txt
fi


deactivate

exit 0
