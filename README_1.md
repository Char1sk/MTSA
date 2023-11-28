# HW1 README

## Requirements

Packages used in this project are as follows. Installing them with Pip or Anaconda is recommended.

- python 3.11.5
- numpy 1.26.0
- matplotlib 3.7.1
- pandas 1.5.3
- brokenaxes

## Datasets

Datasets used in this project are as follows. Download [here](https://box.nju.edu.cn/d/b33a9f73813048b8b00f).

- electricity
- ETT-small
- exchange_rate
- illness
- m4
- traffic
- weather

## Usage

```bash
# To train and evaluate a model on a dataset with a transform:
python ./main.py --job main --dataset <dataset_name> --data_path <datacsv_path> --model <model_name> --transform <trans_name>
# To train and evaluate all models on all datasets with all transforms, and save metrics in a file:
python ./main.py --job test
# To compare KNN with or without LSH approximation on ETTh1 dataset and illness dataset:
python ./main.py --job knn_test
# To test Dataset implementation and data visualization:
python ./src/dataset/dataset.py
# To test Transform implementation:
python ./src/dataset/transforms.py
```

## Reproduce

```bash
# To reproduce Part1 testing: (samples to visualize are randomly selected)
python ./src/dataset/dataset.py
# To reproduce Part4 testing: (test on all models/datasets, need to select the desired ones)
python ./main.py --job test
# To reproduce Part5 testing: (time may vary due to OS and hardwares)
python ./main.py --job knn_test
```

