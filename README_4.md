# HW4 README

## Requirements

Packages used in this project are as follows. Installing them with Pip or Anaconda is recommended.

- python 3.11.5
- numpy 1.26.0
- matplotlib 3.7.1
- pandas 1.5.3
- brokenaxes
- statsmodels 0.14.0
- sklearn

## Datasets

Datasets used in this project are as follows. Download [here](https://box.nju.edu.cn/d/b33a9f73813048b8b00f).

- ETT-small(ETTh1)
- ETT-small(ETTh2)
- ETT-small(ETTm1)
- ETT-small(ETTm2)

## Usage & Reproduce

```bash
# To reproduce Part3 Global Evaluation:
python main.py  --dataset Global --model DLinear --transform StandardizationTransform  --pred_len 96
python main.py  --dataset Global --model TsfKNN --transform StandardizationTransform  --pred_len 96 --n_neighbors <n> --distance <dist>
# To reproduce Part3 SPIRIT Evaluation:
python main.py --dataset ETT --data_path <dataset_path> --model SPIRIT --transform StandardizationTransform --pred_len 96 --individual --n_components <1~7>
```
