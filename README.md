# Bus Detection

## This code is base on other repository

All of this code is adapted/ported/copied from https://github.com/rwightman/efficientdet-pytorch/tree/2fc4e0e107c969090380d0be1a7ad7552dc00c63 

Modification List:
1. Remove unused files and dataset.
2. Adding Bus Dataset (Base on pickle).
3. Remove support of distributed mode.
4. Adding logging via weights and biases

## Run
As first step, you should built the dataset one of the scripts in .scripts 



## Bus Dataset



## TODO
1. Regression Loss
    * GIoU
    * CIoU
    * Mean-Variance estimation
2. Replace BN - change BN due to small batch size on single GPU.
3. Data Augementation
    * Mosic
    * CutPastObject
