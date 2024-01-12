# HW1 README

## Requirements

Packages used in this project are as follows. Installing them with Pip or Anaconda is recommended.

- python 3.11.5
- numpy 1.26.0
- matplotlib 3.7.1
- pandas 1.5.3
- brokenaxes
- statsmodels 0.14.0

## Datasets

Datasets used in this project are as follows. Download [here](https://box.nju.edu.cn/d/b33a9f73813048b8b00f).

- ETT-small(ETTh1)

## Usage

```bash
# To train and evaluate a model on a dataset:
python ./main.py --dataset <dataset_name> --data_path <datacsv_path> --model <model_name> --distance <dist> --knn_tau <lag> --decomp <method>
# To test all distances with lags on TsfKNN, and save results in a file:
python ./main.py --job knn_test_embed_dist
# To test all decompositions on TsfKNN/DLinear, and save results in a file:
python ./main.py --job decomp_test
```

## Reproduce

```bash
# To reproduce Part5.1 testing:
python ./main.py --job knn_test_embed_dist
# To reproduce Part5.2 testing:
python ./main.py --job decomp_test
```
