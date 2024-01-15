# Report

## Part 1. Global-Local Model

### 1.1 Global and Local

本节主要分析GlobalModel和LocalModel的区别。根据PPT上的说法，Global即全局模型，假设不同序列来自于同一个分布，是对多个/全局时间序列上得到一个统一的、共用的模型；而Local即局部模型，是假设序列是由不同的过程生成的，因此使用不同的模型去建模不同的序列。

在本次实验中，实现的Global和Local的区别主要在于是否使用多个数据集对模型进行联合训练，每个数据集可能来自不同的分布/过程。在使用一个模型学习了多个数据集的集成后，它应当能够应用于预测这些数据集中任何一个序列的预测等问题。在本次作业中，选择对ETT-h1、ETT-h2、ETT-m1、ETT-m2进行联合训练。

### 1.2 Global Model

**实现思路**：既然是对数据集的集成，那么只需要实现一个全局数据集即可，然后尽可能保证接口和普通的数据集相同；实在无法保证的，再考虑对特殊情况进行判断，然后执行不同的接口。

**实现细节**：构建一个GlobalDataset，在初始化时即同时初始化并保存了了ETT的4个Dataset：ETT-h1、ETT-h2、ETT-m1、ETT-m2，然后将自身的type设为'Global'，方便进行特殊情况下的判断；对于read_data和split_data函数，不需要任何实现，因为子数据集ETT会自动进行读取和划分。

但问题出现了。训练时Trainer获取的是Dataset的train_data，而这个ndarray是(1 x len x channels)的格式，即数据集的原始序列；但现在我们有多个数据集的多个序列。并且在训练前要先进行transform处理，然后再进行滑窗操作。GlobalDataset不能简单地将多个数据集的多个序列在时间上进行拼接，因为在滑窗时就会造成数据集的混乱。因此，最好是先手动将不同序列进行transform并滑窗之后，将得到的window进行拼接即可，这相当于数据集sample的扩大。因此实现了transform_and_slide函数：

```python
class GlobalDataset(DatasetBase):
    def __init__(self, args) -> None:
        self.datasets = []
        for path in args.global_data_paths:
            args_copy = copy.deepcopy(args)
            args_copy.dataset = 'ETT'
            args_copy.data_path = path
            self.datasets.append(get_dataset(args_copy))
        
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.type = 'Global'
    
    def read_data(self):
        pass
    
    def split_data(self, seq_len):
        pass
    
    def transform_and_slide(self, transform):
        self.train_data = [transform.transform(d.train_data) for d in self.datasets]
        self.val_data   = [transform.transform(d.val_data  ) for d in self.datasets]
        self.test_data  = [transform.transform(d.test_data ) for d in self.datasets]
        self.train_data_wins = np.concatenate([np.concatenate((
            [sliding_window_view(v, (self.seq_len+self.pred_len, d.shape[2])).squeeze(1)
            for v in d])) for d in self.train_data], axis=0)
        self.val_data_wins   = np.concatenate([np.concatenate((
            [sliding_window_view(v, (self.seq_len+self.pred_len, d.shape[2])).squeeze(1)
            for v in d])) for d in self.val_data  ], axis=0)
        self.test_data_wins  = np.concatenate([np.concatenate((
            [sliding_window_view(v, (self.seq_len+self.pred_len, d.shape[2])).squeeze(1)
            for v in d])) for d in self.test_data ], axis=0)
    
```

但此时问题依然存在：一是新的函数不在普通Dataset的接口内，想要完全兼容的话就无法调用；二是GlobalDataset返回的要么是多个数据集序列的列表，要么是多个数据集滑窗的拼接，和普通Dataset返回的单个数据集的序列的形式不符。因此，需要在trainer.py内进行特判处理，利用Dataset的type来进行判断是否是GlobalDataset。在训练时，转而使用滑窗过后的数据直接训练模型；在测试时，对各个数据集的测试集单独进行测试，从而进行公平比较。

```python
class MLTrainer:
    def __init__(self, model, transform, dataset):
        self.model = model
        self.transform = transform
        self.dataset = dataset

    def train(self):
        if self.dataset.type == 'Global':
            self.dataset.transform_and_slide(self.transform)
            train_X_wins = self.dataset.train_data_wins
            self.model.slided_fit(train_X_wins)
        else:
            # normal case, omitted...

    def evaluate(self, dataset, seq_len=96, pred_len=96):
        if dataset.type == 'Global':
            results = []
            for test_data in dataset.test_data:
                test_data = self.transform.transform(test_data)
                subseries = np.concatenate(([sliding_window_view(v, (seq_len + pred_len, v.shape[-1])) for v in test_data]))
                test_X = subseries[:, 0, :seq_len, :]
                test_Y = subseries[:, 0, seq_len:, :]
                te_X = test_X
                fore = self.model.forecast(te_X, pred_len=pred_len)
                # fore = self.transform.inverse_transform(fore)
                mse_result, mae_result = mse(fore, test_Y), mae(fore, test_Y)
                results.append((mse_result, mae_result))
                print('mse:', mse_result)
                print('mae:', mae_result)
            return results
        else:
            # normal case, omitted...
```

## Part 2. SPIRIT

根据PPT上的讲解，SPIRIT方法主要流程为：降维-低维预测-还原维度。

### 2.1 PCA

该方法SPIRIT中所使用的方法为PCA，即**主成分分析**。它是一种常用的数据降维算法，主要思想是将n维特征映射到k维上，这k维是全新的正交特征也被称为主成分，是在原有n维特征的基础上重新构造出来的k维特征。这些k维特征是原始数据特征中对方差(特征)贡献最大的k个，因此往往能够保留住数据最重要的部分，而忽略一些相对不太重要的成分。

PCA的计算流程可以表示为：

1. 数据标准化：$X_c = X - X.mean()$，即使数据本身零均值化；此时也可以除以标准差，从而避免各特征变化范围较大引起的偏差
2. 协方差矩阵：$X_{cov}=\frac1nX_c^T X_c$，计算标准化后数据的协方差矩阵，从而反映各个特征之间的相关性
3. 求解特征值：对协方差矩阵求它的特征值和特征向量
4. 选择主成分：选择特征值较大的k个，并将其对应的特征值进行拼接，得到矩阵$P$
5. 降维映射：$X_{down} = PX$，即可得到降维后的结果

### 2.2 SPIRIT

在完成主成分分析后，降维和逆变换操作都能很容易地进行处理，只需要仅需数据的拟合了，此处根据作业指示选择DLinear方法作为预测模型。主成分分析在该SPIRIT方法中的作用，主要是进行降维和特征提取；例如，将ETT数据集中的7维特征降低为k维特征。这样在进行中间的通道独立的一元预测时，就可以对更重要、更精简的特征进行拟合预测即可，从而获得精确性和运行效率的提升。

![image-20240115204258901](C:\Users\Charisk\AppData\Roaming\Typora\typora-user-images\image-20240115204258901.png)

在拟合时，可以直接对整个数据序列(N_samples, N_channels)进行PCA降维操作，进行变换后，使用压缩后的低维特征序列训练时间序列模型(此处为DLinear)。在预测时，需要对每个window内的测试数据分别进行PCA变换，然后用预测低维特征的模型预测接下来的低维特征，最后再用PCA将预测出的低维特征还原为高维特征，从而完成序列的预测。

```python
class SPIRIT(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.n_components = args.n_components
        self.pca = PCA(self.n_components)
        self.model = DLinear(args) # But with args.individual=True
    
    def _fit(self, X: np.ndarray) -> None:
        Xlow = np.expand_dims(self.pca.fit_transform(X.squeeze(0)), 0)
        self.model.fit(Xlow)
    
    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        Xlow = np.zeros((X.shape[0], X.shape[1], self.n_components))
        for i in range(X.shape[0]):
            Xlow[i,:,:] = self.pca.transform(X[i,:,:])
        Ylow = self.model.forecast(Xlow, pred_len)
        Y = self.pca.inverse_transform(Ylow)
        return Y
```

## Part 3. Evaluation

### 3.1 Global-Local分析

该组实验在预测长度96下进行。

对比是否使用Global的情况，可以发现，DLinear在使用多个数据集的联合数据集的情况下，其评价指标几乎是全面变差了不小的程度，在各数据集上的差异大小各不相同。这可能是由于不同数据集之间仍然存在着分布差异，在对线性模型求解析解时会受到其他无关数据集的干扰，从而导致预测时可能会带有其他分布的偏向而预测，所以指标变差。但对TsfKNN而言，使用联合数据集带来却几乎带来了全面的指标提升，这可能是由于KNN的预测方式不同。由于预测时，TsfKNN只需要寻找最近的几个近邻，并用他们的值进行组合预测即可。如果其他数据中有贴切的数据，就更能够辅助其进行预测；对于无关的数据，TsfKNN会自动将他过滤掉，因此它几乎没有受到负面影响。

另外，可以比较DLinear是否实用通道独立情况的结果。在部分数据集上，使用通道独立方法能够获得或多或少的效果提升，但在部分数据集下，使用通道独立又反而带来的效果的变差，这和HW3中的分析是一致的，即应当谨慎使用通道独立的方法，且需要注意其带来的速度降低。

同时，TsfKNN依然是全面弱于DLinear方法，且预测速度非常慢，因此使用的优先级并不高。

|Method|Global|Independent|ETT-h1|ETT-h1|ETT-h2|ETT-h2|ETT-m1|ETT-m1|ETT-m2|ETT-m2|
| -------------| :----: | :----:  | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
||||MSE|MAE|MSE|MAE|MSE|MAE|MSE|MAE|
|DLinear|yes|no|0.5591|0.5407|0.6794|0.5731|0.6802|0.6061|0.4702|0.5025|
|DLinear|no| no|**0.5262**|**0.5164**|**0.6544**|0.5586|0.6563|0.5899|0.4515|0.4891|
|DLinear|yes|yes|0.5639|0.5465| 0.6857 |0.5769|0.6714|0.6004|0.4609|0.4946|
|DLinear|no| yes|0.5312|0.5181|0.6651|**0.5546**|**0.5923**|**0.5583**| **0.4450** |**0.4814**|
|TsfKNN| yes|no|<u>1.0707</u>|<u>0.7889</u>|<u>1.0888</u>|<u>0.8010</u>|<u>0.8745</u>| <u>0.7194</u> |<u>0.8799</u>|<u>0.7163</u>|
|TsfKNN| no| no|1.1005|0.7961|1.2259|0.8344|0.8920|0.7217|0.9991|0.7590|

### 3.2 SPIRIT分析

该组实验在ETT-h1下进行，预测长度为96，使用通道独立。

可以看到，随着选取的主成分个数的增加，模型的指标也在不断地变好，且在收敛前其变化情况是几乎线性的。而当主成分达到5个时，此时的指标几乎和不使用PCA(不用SPIRIT而是直接DLinear)时的效果相差无几，具体表现为MSE稍低而MAE稍高。在此以后，继续增加主成分个数不会再带来显著的效果提升，只会出现非常微小的指标提升。

因此，在使用诸如PCA的降维方法时，需要对降低到的维数进行谨慎地选取，从而避免造成较大而性能损失，从而在保持或超越模型效果的同时能够加速其过程。


| Components | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 0(NoPCA) |
| :------: | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|    MSE    |0.891|0.75|0.6781|0.6292|0.5306|0.5301|0.5301|0.5313|
|    MAE    |0.7247|0.6568|0.6092|0.5769|0.5204|0.52|0.5199|0.5181|


![image-20240115212827627](C:\Users\Charisk\AppData\Roaming\Typora\typora-user-images\image-20240115212827627.png)

![image-20240115212745142](C:\Users\Charisk\AppData\Roaming\Typora\typora-user-images\image-20240115212745142.png)
