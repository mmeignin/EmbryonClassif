# EmbryonClassif
Deep learning classification of bovine embryos

It includes :
- Classification Model : LitClassificationModel divided in [LitBakbone ,LitHead]
- Training and inferrence for this model

## Getting started

In the Build dir you can run on cluster your model you have to speciy first the inputs like the model parameters and hyper-parameters as well as the datasets.

## Environment Versioning

In the Build dir : We are creating an environment in order to be consistent in the versioning of the requirements. <br />
You can find Requirements without Version Specifiers and Requirements with Version Specifiers in Build/requirements.txt. <br />

## Datasets

You can find some created Datasets in DataSplit and NewDataSplit:<br />
&nbsp;&nbsp;&nbsp;&nbsp;DataSplit:<br />
- Embryon_RandomSplit : train,test,val : 180, 46, 47 elements , class repartition is the same in all three sets <br />
- Embryon_RandomSpliRaw : same as the first one but for non-segmented videos <br />
- Embryon_RawAndSegmented : datasets with same repartition as the previous one but with both non-segmented and segmented videos <br />
- EmbryonBinary_RandomSplit : same repartition but for binary classification , viable classification <br />
- EmbryonBinaryRaw_RandomSplit : same as the previous one but for raw videos <br />
- transferable : same repartition but for transferable classification <br />
- transferable_Raw : same as the previous one but for raw videos <br />
NewDataSplit: <br />
- Binary_FV : class repartition is fifty-fifty in validation set otherwise it is equal to data-class repartition, train,test,val : 177,56,40 elements <br />
- Embryon_RandomSplit : same as DataSplit/Embryon_RandomSplit but with different random seed <br />
- transferable_FV : class repartition is fifty-fifty in validation set otherwise it is equal to data-class repartition, train,test,val : 177,56,40 elements <br />
- Embryon_RS_fixed_val :class repartition is even in validation set otherwise it is equal to data-class repartition, train,test,val : 177,56,40 elements <br />


## Models

Different Model proposition that can be used for classification:<br />
features extractor :<br />
- ResNet18<br />
-  ResNet34<br />
- SimpleConv<br />
head:<br />
- Gru<br />
- Lstm<br />
- ConvPooling<br />
- NaiveHead<br />

## Dataloaders

csvflowdatamodule is divided into transforms, where one can find all the transform that we considered for the project, brightness adjustment, flips and some for segmented videos and into CsvDataset that contains tools in order to use videos as inputs for our machine learning project efficiently


## Notebooks

The Notebooks dir contains the tools used for confusion matrix creation and datasplit creation.

## csvfiles

It contains all results from previous training for analysis











