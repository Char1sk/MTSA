from src.models.TsfKNN import TsfKNN
from src.models.DLinear import DLinear
from src.models.ARIMA import ARIMA
from src.models.ThetaMethod import ThetaMethod
from src.models.ResidualModel import ResidualModel
from src.models.baselines import ZeroForecast, MeanForecast, LinearRegression, ExponantialSmoothing
from src.utils.transforms import IdentityTransform, NormalizationTransform, StandardizationTransform, MeanNormalizationTransform, BoxCoxTransform
from trainer import MLTrainer
from src.dataset.dataset import get_dataset
import argparse
import random
import time
import numpy as np


def get_model(args):
    model_dict = {
        'ZeroForecast': ZeroForecast,
        'MeanForecast': MeanForecast,
        'LinearRegression': LinearRegression,
        'ExponantielSmoothing': ExponantialSmoothing,
        'TsfKNN': TsfKNN,
        'DLinear': DLinear,
        'ARIMA': ARIMA,
        'ThetaMethod': ThetaMethod,
        'ResidualModel': ResidualModel
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
    # parser.add_argument('--data_path', type=str, default='./dataset/ETT-small/ETTh1.csv')
    parser.add_argument('--data_path', type=str, default='./dataset/illness/national_illness.csv')
    parser.add_argument('--train_data_path', type=str, default='./dataset/m4/Daily-train.csv')
    parser.add_argument('--test_data_path', type=str, default='./dataset/m4/Daily-test.csv')
    parser.add_argument('--dataset', type=str, default='ETT', help='dataset type, options: [M4, ETT, Custom]')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--frequency', type=str, default='h', help='frequency of time series data, options: [h, m]')
    
    # forcast task config
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=64, help='prediction sequence length')
    parser.add_argument('--es_lambda', type=float, default=0.2, help='hyper-parameter lambda in ES')
    
    # model define
    # parser.add_argument('--model', type=str, required=True, default='MeanForecast', help='model name')
    parser.add_argument('--model', type=str, default='MeanForecast', help='model name')
    parser.add_argument('--n_neighbors', type=int, default=1, help='number of neighbors used in TsfKNN')
    parser.add_argument('--distance', type=str, default='euclidean', help='distance used in TsfKNN')
    parser.add_argument('--msas', type=str, default='MIMO', help='multi-step ahead strategy used in TsfKNN, options: '
                                                                 '[MIMO, recursive]')
    parser.add_argument('--individual', action='store_true', help='use individual weights for Linear & DLinear')
    # KNN approximate LSH
    parser.add_argument('--knn_variant', type=str, default=None, help='KNN variants, options: [None, "LSH", "Decomp]')
    parser.add_argument('--hash_size', type=int, default=8, help='number of digits after hash in LSH')
    parser.add_argument('--num_hashes', type=int, default=1, help='number of hash tables in LSH')
    # KNN lag-based
    parser.add_argument('--knn_tau', type=int, default=1, help='stride of lag-based embedding')
    parser.add_argument('--knn_m', type=int, default=0, help='dim of embedding, no greater than seq_len; 0 means all')
    # transform define
    parser.add_argument('--transform', type=str, default='IdentityTransform')
    parser.add_argument('--boxcox_lambda', type=float, default=2.0, help='hyper-parameter lambda in BoxCox')
    # decomp function
    parser.add_argument('--decomp', type=str, default='moving_average', help='["moving_average", "differential_decomposition", "STL_decomposition", "X11_decomposition"]')
    # job
    parser.add_argument('--job', type=str, default='main', help='select in ["main", "test", "knn_test", "knn_test_embed_dist", "decomp_test"]')
    # ResidualModel
    parser.add_argument('--residual', action='store_true', help='use residual for decomposition in ResidualModel')
    parser.add_argument('--residual_mode', type=str, default='season_first', help='[trend_first|season_first]')
    parser.add_argument('--model_t', type=str, default='LinearRegression', help='model for Trend')
    parser.add_argument('--model_s', type=str, default='LinearRegression', help='model for Season')
    parser.add_argument('--model_r', type=str, default='LinearRegression', help='model for Residual')
    # GlobalDataset
    parser.add_argument('--global', action='store_true', help='use Global Dataset for Model')
    parser.add_argument('--global_data_paths', type=list, default=["./dataset/ETT-small/ETTh1.csv", "./dataset/ETT-small/ETTh2.csv", "./dataset/ETT-small/ETTm1.csv", "./dataset/ETT-small/ETTm2.csv"], help='datasets')
    
    
    args = parser.parse_args()
    return args


def set_args(args, datapath, model, pred_len):
    args.data_path = datapath
    args.model = model
    args.pred_len = pred_len


def main(args):
    # load dataset
    dataset = get_dataset(args)
    # create model
    model = get_model(args)
    # data transform
    transform = get_transform(args)
    # create trainer
    trainer = MLTrainer(model=model, transform=transform, dataset=dataset)
    
    time1 = time.time()
    # train model
    trainer.train()
    # evaluate model
    trainer.evaluate(dataset, seq_len=args.seq_len, pred_len=args.pred_len)
    
    time2 = time.time()
    print(f"time: {time2-time1}")


def test(args):
    fix_seed = 2023
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    
    datapaths = {
        'ETTh1': './dataset/ETT-small/ETTh1.csv',
        'ETTh2': './dataset/ETT-small/ETTh2.csv',
        'ETTm1': './dataset/ETT-small/ETTm1.csv',
        'ETTm2': './dataset/ETT-small/ETTm2.csv',
    }
    models = ['ResidualModel']
    pred_lens = [96, 192, 336, 720]
    
    
    with open('./results.csv', 'a') as f:
        f.write('datapaths, models, pred_lens, mse, mae\n')
        for dp in datapaths.keys():
            for md in models:
                for pl in pred_lens:
                    random.seed(fix_seed)
                    np.random.seed(fix_seed)
                    
                    print(dp,md,pl)
                    set_args(args, datapaths[dp], md, pl)
                    dataset = get_dataset(args)
                    model = get_model(args)
                    transform = get_transform(args)
                    trainer = MLTrainer(model=model, transform=transform, dataset=dataset)
                    trainer.train()
                    results = trainer.evaluate(dataset, seq_len=args.seq_len, pred_len=args.pred_len)
                    line = [dp, md, str(pl)] + [f'{r:.4}' for r in results]
                    f.write(','.join(line)+'\n')


def knn_test(args):
    fix_seed = 2023
    
    np.set_printoptions(linewidth=200)
    for (dataset, datapath) in [('Custom', './dataset/illness/national_illness.csv'), ('ETT', './dataset/ETT-small/ETTh1.csv')]:
        # set_args(args, dataset, datapath, 'BoxCoxTransform', 'TsfKNN')
        set_args(args, dataset, datapath, 'IdentityTransform', 'TsfKNN')
        for approx in [None, 'LSH']:
            args.approx = approx
            dataset = get_dataset(args)
            model = get_model(args)
            transform = get_transform(args)
            # Get time and results
            results_list = []
            times_list = []
            rounds = 5
            for i in range(rounds+1):
                random.seed(fix_seed)
                np.random.seed(fix_seed)
                # Dont count in 1st time without cache
                trainer = MLTrainer(model=model, transform=transform, dataset=dataset)
                if i != 0:
                    time1 = time.time()
                trainer.train()
                results = trainer.evaluate(dataset, seq_len=args.seq_len, pred_len=args.pred_len)
                if i != 0:
                    time2 = time.time()
                if i != 0:
                    times_list.append(time2-time1)
                    results_list.append(results)
            # Calculate time and results
            results_mean = np.array(results_list).mean(axis=0)
            times_mean = np.array(times_list).mean(axis=0)
            # Print results
            print(datapath, approx)
            print(f'\t{results_mean}, {times_mean}s')


def knn_test_embed_dist(args):
    fix_seed = 2023
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    
    taus = [1,2,3,4,5]
    dists = ['euclidean', 'manhattan', 'chebyshev', 'cosine']
    args.model = 'TsfKNN'
    args.knn_m = 0
    args.n_neighbors = 3
    args.transform = 'StandardizationTransform'
    with open('./results.csv', 'w') as f:
        for d in dists:
            for t in taus:
                print(d, t)
                args.knn_tau = t
                args.distance = d
                dataset = get_dataset(args)
                model = get_model(args)
                transform = get_transform(args)
                trainer = MLTrainer(model=model, transform=transform, dataset=dataset)
                trainer.train()
                rmse, rmae = trainer.evaluate(dataset, seq_len=args.seq_len, pred_len=args.pred_len)
                f.write(f"{d}, {t}, {rmse}, {rmae}\n")


def decomp_test(args):
    fix_seed = 2023
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    
    args.transform = 'StandardizationTransform'
    # args.model = 'TsfKNN'
    # args.knn_variant = 'Decomp'
    # args.n_neighbors = 3
    args.model = 'DLinear'
    decomps = ["moving_average", "differential_decomposition", "STL_decomposition"]
    with open('./results.csv', 'w') as f:
        for dp in decomps:
            print(args.model, dp)
            args.decomp = dp
            args.knn_variant = 'Decomp'
            dataset = get_dataset(args)
            model = get_model(args)
            transform = get_transform(args)
            trainer = MLTrainer(model=model, transform=transform, dataset=dataset)
            trainer.train()
            rmse, rmae = trainer.evaluate(dataset, seq_len=args.seq_len, pred_len=args.pred_len)
            f.write(f"{args.model}, {dp}, {rmse}, {rmae}\n")


if __name__ == '__main__':
    jobs = {
        'main': main,
        'test': test,
        'knn_test': knn_test,
        'knn_test_embed_dist': knn_test_embed_dist,
        'decomp_test': decomp_test
    }
    args = get_args()
    jobs[args.job](args)
    # main(args)
    # test(args)
    # knn_test(args)
