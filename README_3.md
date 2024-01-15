# HW3 README

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
- ETT-small(ETTh2)
- ETT-small(ETTm1)
- ETT-small(ETTm2)

## Usage

```bash
# To train and evaluate a model on one dataset:
python ./main.py --dataset ETT --data_path <datacsv_path> --model <ARIMA|ThetaMethod> --decomp <STL|X11>
```

## Reproduce

```bash
# To reproduce Part4 ResidualModel Evaluation:
python main.py --job test --dataset ETT --model ResidualModel --transform StandardizationTransform --decomp X11_decomposition --model_t LinearRegression --model_s LinearRegression --model_r <MeanForecast|LinearRegression> [--residual --residual_mode <season_first|trend_first>] [--individual]
```
