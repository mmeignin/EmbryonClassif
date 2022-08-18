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
        python3 training_hb.py  \
                                --data_file EmbryonBinaryRaw_RandomSplit\
                                --criterion_name bce_balanced\
                                -bb SimpleConv -he ConvPooling\
                                #--augmentation randombrightness \
                                #--preload_cache #\
                                #-pb
                                
else 
        echo "Virtual Environment issue, env name: ${$VIRTUAL_ENV}"
fi
##Check missing requirements
#pip freeze > virtual_env_requirements.txt
deactivate

exit 0
