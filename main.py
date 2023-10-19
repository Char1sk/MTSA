from src.models.TsfKNN import TsfKNN
from src.models.baselines import ZeroForecast, MeanForecast, LinearRegression, ExponantielSmoothing
from src.utils.transforms import IdentityTransform, NormalizationTransform, StandardizationTransform, MeanNormalizationTransform, BoxCoxTransform
from trainer import MLTrainer
from src.dataset.dataset import get_dataset
import argparse
import random
import numpy as np


def get_model(args):
    model_dict = {
        'ZeroForecast': ZeroForecast,
        'MeanForecast': MeanForecast,
        'LinearRegression': LinearRegression,
        'ExponantielSmoothing': ExponantielSmoothing,
        'TsfKNN': TsfKNN,
    }
    return model_dict[args.model](args)


def get_transform(args):
    transform_dict = {
        'IdentityTransform': IdentityTransform,
        'NormalizationTransform': NormalizationTransform,
        'StandardizationTransform': StandardizationTransform,
        'MeanNormalizationTransform': MeanNormalizationTransform,
        'BoxCoxTransform': BoxCoxTransform
    }
    return transform_dict[args.transform](args)


def get_args():
    parser = argparse.ArgumentParser()

    # dataset config
    # parser.add_argument('--data_path', type=str, default='./dataset/ETT/ETTh1.csv')
    parser.add_argument('--data_path', type=str, default='./dataset/ETT-small/ETTh1.csv')
    parser.add_argument('--train_data_path', type=str, default='./dataset/m4/Daily-train.csv')
    parser.add_argument('--test_data_path', type=str, default='./dataset/m4/Daily-test.csv')
    parser.add_argument('--dataset', type=str, default='ETT', help='dataset type, options: [M4, ETT, Custom]')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--ratio_train', type=int, default=0.7, help='train dataset length')
    parser.add_argument('--ratio_val', type=int, default=0, help='validate dataset length')
    parser.add_argument('--ratio_test', type=int, default=0.3, help='input sequence length')

    # forcast task config
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=32, help='prediction sequence length')
    parser.add_argument('--es_lambda', type=float, default=0.2, help='hyper-parameter lambda in ES')

    # model define
    # parser.add_argument('--model', type=str, required=True, default='MeanForecast', help='model name')
    parser.add_argument('--model', type=str, default='MeanForecast', help='model name')
    parser.add_argument('--n_neighbors', type=int, default=1, help='number of neighbors used in TsfKNN')
    parser.add_argument('--distance', type=str, default='euclidean', help='distance used in TsfKNN')
    parser.add_argument('--msas', type=str, default='MIMO', help='multi-step ahead strategy used in TsfKNN, options: '
                                                                 '[MIMO, recursive]')

    # transform define
    parser.add_argument('--transform', type=str, default='IdentityTransform')
    parser.add_argument('--boxcox_lambda', type=float, default=1.0, help='hyper-parameter lambda in BoxCox')

    args = parser.parse_args()
    return args


def set_args(args, dataset, datapath, transform, model):
    args.dataset = dataset
    args.data_path = datapath
    args.transform = transform
    args.model = model


def main():
    fix_seed = 2023
    random.seed(fix_seed)
    np.random.seed(fix_seed)

    args = get_args()
    # load dataset
    dataset = get_dataset(args)
    # create model
    model = get_model(args)
    # data transform
    transform = get_transform(args)
    # create trainer
    trainer = MLTrainer(model=model, transform=transform, dataset=dataset)
    # train model
    trainer.train()
    # evaluate model
    trainer.evaluate(dataset, seq_len=args.seq_len, pred_len=args.pred_len)


def test():
    fix_seed = 2023
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    args = get_args()
    
    datasets = ['M4', 'ETT', 'Custom']
    datapaths = {
        'electricity': './dataset/electricity/electricity.csv',
        'exchange_rate': './dataset/exchange_rate/exchange_rate.csv',
        'illness': './dataset/illness/national_illness.csv',
        'traffic': './dataset/traffic/traffic.csv',
        'weather': './dataset/weather/weather.csv'
    }
    models = ['ZeroForecast','MeanForecast','LinearRegression','ExponantielSmoothing']
    transforms = [
        'IdentityTransform',
        'NormalizationTransform',
        'StandardizationTransform',
        'MeanNormalizationTransform',
        'BoxCoxTransform'
    ]
    
    with open('./results.csv', 'w') as f:
        for ds in datasets:
            if ds != 'Custom':
                continue
            for dp in datapaths.keys():
                for md in models:
                    for tf in transforms:
                        print(dp,md,tf)
                        set_args(args, ds, datapaths[dp], tf, md)
                        dataset = get_dataset(args)
                        model = get_model(args)
                        transform = get_transform(args)
                        trainer = MLTrainer(model=model, transform=transform, dataset=dataset)
                        trainer.train()
                        results = trainer.evaluate(dataset, seq_len=args.seq_len, pred_len=args.pred_len)
                        line = [dp, md, tf] + [str(r) for r in results]
                        f.write(','.join(line)+'\n')
                    


if __name__ == '__main__':
    # main()
    test()
