# PAPER NAME

This repository is for the paper PAPER NAME.
This is a fork project from: https://github.com/D-X-Y/AutoDL-Projects
Therefore, system setup requirements are the same.

## Generating Proxy Datasets
Generating a proxy dataset (e.g., random search with r={0.75, 0.5, 0.25}, r-values are preset in script):
```
python rs_pre.py
```

Available generator scripts:
+ rs_pre.py
+ ccrs_pre.py
+ kmeans_removal.py
+ cc_outlier_removal.py
+ ae_removal.py 
+ tlresnet18.py

## Cell Search
Running a Cell Search on a sampled Proxy Dataset (e.g., DARTS-V2 on sampled with AE-Easy and r=0.25):

```
bash ./scripts-search/algos/DARTS-V2.sh cifar100 1 -1 aeeasy 0.25
```

Possible proxy dataset parameters are:
+ rs
+ ccrs
+ tleasy
+ tlhard
+ aeeasy
+ aehard
+ cc_outlier_removal
+ kmeans_50_removal
+ kmeans_100_removal
+ kmeans_150_removal
+ kmeans_200_removal
+ kmeans_dc_50_removal
+ kmeans_dc_100_removal
+ kmeans_dc_150_removal
+ kmeans_dc_200_removal
+ kmeans_dc_removal_v2
+ kmeans_dc_removal_v2_raw

For the paper, following search scripts are used:
+ ./scripts-search/algos/DARTS-V1.sh
+ ./scripts-search/algos/DARTS-V2.sh
+ ./scripts-search/algos/ENAS.sh
+ ./scripts-search/algos/GDAS.sh

## Cell Evaluation

Running the Cell Evaluation requires the following line:

```
bash ./scripts-search/NAS-Bench-201/train-a-net.sh network_design channels num_cells
```

where channels are 16, 32, 64, and num_cells is 5. The network_design is the output from the Cell Search and can have the form:
```
|nor_conv_3x3~0|+|nor_conv_3x3~0|none~1|+|nor_conv_3x3~0|none~1|none~2|+|avg_pool_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|nor_conv_3x3~3|
```
