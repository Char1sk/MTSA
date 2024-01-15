# Report

## Part 1. Decomposition

**前置**：对于本次分解任务，与上一次的分解不同，这次分解主要是分解为趋势项、季节项和剩余项；由于之前实现的分解方式为加性分解，即满足：$Y=T+S+R$，而非乘性分解$Y=TSR$，因此此处实现的STL和X11也为加性分解，从而能够进行更公平的比较分析。

### 1.1 STL Decomposition

**STL分解**：即Seasonal-trend decomposition using LOWESS，该方法使用LOWESS做局部加权回归，使序列平滑化，获得趋势项。在获得趋势项后，对去趋势项以季节周期为步长取平均值，将该平均值作为对应位置的季节项。然后再对此进行多次迭代，获得最终的分解结果。该种分解已在上次作业中实现过，此处不再赘述。

**实现细节**：本次实现对上次的实现方式做了更改。上次实现使用statsmodels库的lowess函数来做局部加权回归作为趋势项，再手动计算季节项和剩余项，并且没有进行迭代。本次实现直接使用statsmodels的STL方法。同样由于STL仅能处理1d的向量，因此实际分解时需要对所有的sample、所有的channel分别进行STL分解，效率相当低。

```python
X_trend = np.zeros_like(X)
X_season = np.zeros_like(X)
X_residual = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[2]):
        result = STL(X[i,:,j], period=seasonal_period, robust=True).fit()
        X_trend[i,:,j] = np.array(result.trend)
        X_season[i,:,j] = np.array(result.seasonal)
        X_residual[i,:,j] = np.array(result.resid)

return (X_trend, X_season, X_residual)
```

### 1.2 X11 Decomposition

**X11分解**：X11分解流程主要参考PPT讲义及[链接]([X-11 (jdemetradocumentation.github.io)](https://jdemetradocumentation.github.io/JDemetra-documentation/pages/theory/SA_X11.html))。该方法首先使用复合移动平均得到趋势项$T^1$，然后从原序列中剔除趋势项得到$deT^1$，由之计算得到季节项$S^1$，再从原序列剔除季节项得到$deS^1$，对之使用Henderson移动平均重新得到趋势项$T^2$（此处PPT中书写错误，$Y^2$和$X^2$书写混乱）。之后，重新使用该流程得到$deT^2,S^2,deS^2,T^3$，最后从去季节项中去除趋势项，得到剩余项$R$。（对于PPT中的乘性分解，去除指除法；对于实现中的加性分解，去除指减去）

![image-20240115153121381](Report_HW3.assets/image-20240115153121381.png)

需要注意的是，本次实现的X11分解为加性分解，从而和之前所实现的其他分解方法形成对照。此外，两个参考中均含有一些Magic Number，包括$M_{2\times 12}$移动平均、$Henderson_{13}$移动平均等，针对的是周期等于12(即月度数据)的情况。在本次实验中主要使用ETT数据集，其中周期等于24(即以小时为单位)，因此此处改为使用$M_{2\times 24}$移动平均、$Henderson_{23}$移动平均等。针对一些常用的$M_{3\times 3}$等，此处不做更改。

**实现细节**：针对$M_{2\times12}$这样的复合移动平均的实现方式，如同定义一样先后使用了两个移动平均$M_{12}$和$M_2$，此处借用此前实现的滑动平均分解函数。实现Henderson移动平均时，直接使用了其宽度为23的kernel，并使用scipy库的卷积函数实现加权的移动平均；由于卷积在边界会遇到值缺失问题，此处直接使用等效0填充。由于X11分解是支持并行化的，因此运行速度上比STL快很多，但比移动平均分解稍慢。

```python
# Step 01: Trend T1
T1 = moving_average(moving_average(X, seasonal_period)[0], 2)[0]
# Step 02: DeTrend DT1 (S,I)
DT1 = X - T1
# Step 03: Season S1
S1 = moving_average(moving_average(DT1, 3)[0], 3)[0] - moving_average(moving_average(DT1, seasonal_period)[0], 2)[0]
# Step 04: DeSeason DS1 (T,I)
DS1 = X - S1
# Step 05: Trend T2
Hw23 = np.array((-0.004, -0.011, -0.016, -0.015, -0.005, 0.013, 0.039, 0.068, 0.097, 0.122, 0.138, 0.148, 0.138, 0.122, 0.097, 0.068, 0.039, 0.013, -0.005, -0.015, -0.016, -0.011, -0.004))
Hw23 = np.expand_dims(np.expand_dims(Hw23, 0), 2)
T2 = sp.signal.convolve(DS1, np.flip(Hw23), 'same')
# Step 06: DeTrend DT2
DT2 = X - T2
# Step 07: Season S2
S2 = moving_average(moving_average(DT2, 3)[0], 3)[0] - moving_average(moving_average(DT2, seasonal_period)[0], 2)[0]
# Step 08: DeSeason DS2 (T,I)
DS2 = X - S2
# Step 09: Trend T3
T3 = sp.signal.convolve(DS2, np.flip(Hw23), 'same')
# Step 10: I
I = DS2 - T3
    
return (T3, S2, I)
```

## Part 2. Model

### 2.1 ARIMA

**ARIMA**：自回归差分移动平均模型，是对差分后的序列使用ARMA模型进行建模，即$Y_t = c+\phi_1Y_{t-1}+\phi_2Y_{t-2}+\dots+\phi_pY_{t-p}+\theta_1\epsilon_{t-1}+\theta_2\epsilon_{t-2}+\dots++\theta_q\epsilon_{t-q}+\epsilon_t$。此处实现使用了statsmodels库的ARIMA实现。需要注意的是，ARIMA模型同时只能对一个一元的序列进行计算，因此在拟合时需要对不同channel (feature)分别用不同的ARIMA模型进行拟合。在预测时，同样也需要对每一个sample、每一个channel分别进行预测，因此效率相对较低。

另外，本次实现是对训练数据整体进行了一次拟合，然后在测试时进行应用，即Global方式；而非在测试时直接对不同的数据进行拟合、然后直接进行预测。

```python
class ARIMA(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.models = []
    
    def _fit(self, X: np.ndarray) -> None:
        # X: ndarray, (1, time, feature/OT)
        if not self.fitted:
            for i in range(X.shape[2]):
                self.models.append(statsmodels.tsa.arima.model.ARIMA(X[0,:,i]).fit())
    
    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        # X: ndarray, (windows_test, seq_len, features)
        Y = np.zeros((X.shape[0], pred_len, X.shape[2]))
        for j in range(X.shape[2]):
            model = self.models[j]
            for i in range(X.shape[0]):
                Y[i,:,j] = model.apply(X[i,:,j]).forecast(pred_len)
        return Y
```

### 2.2 ThetaMethod

**ThetaMethod**：即对原始序列进行重构，修正其二阶导数以获得Theta-Line，该Theta-Line可以直接得到闭式解，并且还满足性质$\frac12(z_{t,\theta}+z_{t,2-\theta})=y_t$。因此，分别对现有序列构建两条$\theta=0,\theta=2$的Theta-Line，而后者需要用ETS进行估计。最后可得预测方式为$\hat y_n(h) = \tilde y_n(h) + \frac 12 \hat b_0 (h-1+ 1/\alpha)$。

同样的，ThetaMethod也只能对一元序列进行拟合和预测，因此需要对每一个sample、每一个channel都要进行拟合和预测，因而效率稍低。并且，此处使用的是每次对测试数据再进行拟合和预测，即Local方式。

```python
class ThetaMethod(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.models = []
    
    def _fit(self, X: np.ndarray) -> None:
        # X: ndarray, (1, time, feature/OT)
        pass
    
    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        # X: ndarray, (windows_test, seq_len, features)
        Y = np.zeros((X.shape[0], pred_len, X.shape[2]))
        for j in range(X.shape[2]):
            for i in range(X.shape[0]):
                Y[i,:,j] = ThetaModel(X[i,:,j], period=24).fit().forecast(pred_len)
        return Y
```

## Part 3. Residual Model

本次**ResidualModel**针对**分解**、**残差学习**、**通道独立**这三方面集成了不同模型。

**分解**：该模型首先使用实现的X11分解将输入序列分解为趋势项、季节项和剩余项。由于线性模型对季节项的拟合效果较好，因此针对季节项使用线性回归LinearRegression进行拟合和预测。针对趋势项，由于线性模型对趋势项的拟合效果稍弱，因此本意寻找一个稍好的模型对趋势项进行拟合，但在目前已经实现的方法中，暂时还未出现在实验中超越LinearRegression的，因此依然使用了线性回归模型；TsfKNN、ARIMA、ThetaMethod等方法耗时过长，且效果也不佳，因此没有选用。而对于剩余项，由于其通常波动较大且在一定范围内进行波动，此处可以根据奥卡姆剃刀原则考虑使用MeanForecast对其进行预测，而可能可以不用采用另一个线性模型。

**残差学习**：此处涉及HW2中所述的不同DLinear实现版本，因为没有使用神经网络迭代求解而是直接对线性模型求解析解，因此对序列的不同分解顺序产生了不同实现版本。针对非残差实现，此处使用的是实现3，即先对序列滑窗，然后划分XY，再分别对其进行分解，最后分别求解模型；这种实现符合测试时对有限序列长度进行分解时的情形，效果也相对较好。针对残差实现，此处使用的是实现4，即训练时先对序列滑窗，然后对窗口内部的X进行分解，然后只使用趋势项(季节项)作为输入，将目标Y作为标签直接拟合，并将拟合结果作为对目标Y的趋势项(季节项)预测；然后将季节项(趋势项)作为输入、残差Y-Y_trend作为标签进行拟合，将拟合结果作为季节项(趋势项)预测；最后再用剩余项对残差进行拟合。从实验中可得知，是否使用残差在不同数据集下各有优劣。

**通道独立**：在通道独立的情况下，对序列的每一个Channel(feature)单独使用模型进行拟合和预测；而非在通道依赖的情况下，将序列每个时间点的Channel作为该时间点的特征，用一个模型进行统一的拟合和预测。通道独立的方式可能会破坏不同通道间的相关性，但也有了对不同通道进行拟合的自由度，也因此有可能获得更好的效果；但由于是单独对不同通道进行计算，所以速度会有所下降。

## Part 4. Evaluation

### 4.1 分解模型分析

该组实验在没有使用残差学习、使用通道依赖的情况下进行。

由表可以看出，在对X11分解的剩余项进行拟合时，使用线性模型LinearRegression比使用MeanForecast的效果普遍更好，但更好的程度有限。在ETT-h1上，线性模型好的程度最高；而在其他数据集上，其高的程度就不算太高了。另外，随着预测序列的加长，MeanForecase的效果已经逐渐靠近线性模型，也说明剩余项长期来看是在某个范围内波动的序列。因此，使用MeanForecast对剩余项进行预测是可行的，更简洁且效率更高，但可能会带来一定的精度损失。

| Residual Term | pred_len | ETT-h1 | ETT-h1 | ETT-h2 | ETT-h2 | ETT-m1 | ETT-m1 | ETT-m2 | ETT-m2 |
| ------------- | :------: | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  |          | MSE    | MAE    | MSE    | MAE    | MSE    | MAE    | MSE    | MAE    |
| MeanForecast  |    96    |0.5361|0.5286|0.6575|0.5611|0.6612|0.5945|0.4541|0.4923|
| **LinearRegress** |    **96**    |**0.5264**|**0.5167**|**0.6541**|**0.5585**|**0.6561**|**0.59**|**0.4516**|**0.4892**|
| MeanForecast  |   192    |0.5999|0.562|0.8034|0.6304|0.6736|0.6084|0.5634|0.5485|
| **LinearRegress** |  **192** |**0.5918**|**0.5518**|**0.7993**|**0.6285**|**0.6696**|**0.6057**|**0.5618**|**0.5468**|
| MeanForecast  |   336    |0.6485|0.5841|0.88|0.6741|0.738|0.6458|0.744|0.6274|
| **LinearRegress** |  **336** |**0.6413**|**0.5746**|**0.8754**|**0.6722**|**0.7343**|**0.644**|**0.7411**|**0.6255**|
| MeanForecast  |   720    |0.6639|0.5972|0.8916|0.6908|0.8049|0.6892|0.8494|0.6916|
| **LinearRegress** |  **720** |**0.6573**|**0.5883**|**0.8876**|**0.6892**|**0.8011**|**0.6878**|**0.8429**|**0.6889**|

### 4.2 残差模型分析

该组实验在使用线性模型拟合剩余项、使用通道依赖的情况下进行。

表中的trend_first和season_first分别指先用趋势/季节项拟合，再用季节/趋势项拟合残差。起初的猜想是，先用擅长拟合季节项的线性模型拟合数据，再用模型去拟合残差趋势项，可能表现效果会更好。然而从表中可以发现，无论在何种pred_len、何种数据集、何种指标下，二者的指标值几乎没有差异(在舍入前，只有十位之后有微小差异，可认为是浮点误差)。这可能是因为，在使用纯线性模型进行加性分解的结果进行拟合时，由于模型本身是线性的，因此求出解析解后进行相加的这种线性运算对求解的顺序不敏感，因此才会导致结果相同。

另外，对比使用残差模型的结果和4.1节没有使用残差的结果，可以发现使用残差模型时结果会有非常微小的提升，并且这样提升的效果也会随着pred_len的增大而减小。可以猜想残差模型能够对标签进行更彻底的拟合，而非对标签手动的分解结果进行分别的、不彻底的拟合。


| Residual order | pred_len | ETT-h1 | ETT-h1 | ETT-h2 | ETT-h2 | ETT-m1 | ETT-m1 | ETT-m2 | ETT-m2 |
| ------------- | :------: | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  |          | MSE    | MAE    | MSE    | MAE    | MSE    | MAE    | MSE    | MAE    |
|trend_first |    96    |0.5262|0.5165|0.6541|0.5586|0.6559|0.5899|0.4516|0.4893|
|season_first|    96    |0.5262|0.5165|0.6541|0.5586|<u>0.6559</u>|0.5899|0.4516|0.4893|
|trend_first |   192    |0.5915|0.5516|0.7992|0.6284|0.6696|0.6056|0.5618|0.5468|
|season_first|   192    |0.5915|0.5516|0.7992|0.6284|<u>0.6696</u>|0.6056|0.5618|0.5468|
|trend_first |   336    |0.6411|0.5744|0.8753|0.6722|0.734|0.6439|0.7406|0.6254|
|season_first|   336    |0.6411|0.5744|0.8753|0.6722|<u>0.734</u>|0.6439|0.7406|0.6254|
|trend_first |   720    |0.6572|0.5882|0.8875|0.6892|0.801|0.6878|0.8428|0.6888|
|season_first|   720    |0.6572|0.5882|0.8875|0.6892|<u>0.801</u>|0.6878|0.8428|0.6888|

### 4.3 通道独立分析

该组实验在使用残差模型、使用通道独立的情况下进行。

可以看出，相比4.1不使用残差模型、使用通道依赖的情况，本节使用残差模型、使用通道依赖的时，线性模型比MeanForecast效果更好的情况依然存在，且依然保持着pred_len越大、优势越小的特征。

同时可以纵向对比4.2节没有使用通道独立时，使用通道独立的情况下在ETT-m1数据集下效果有较高的提升(见下划线)，但在预测序列足够长时也衰减严重或被反超；在ETT-h1和ETT-h2上反而导致了结果的变差。这种指标的反映和PPT中的非常大的效果提升不完全相同。此外，在数据量较大的情况下，使用通道独立进行预测会导致时间的大量增长，使用的时候需要斟酌。

| Residual Term | pred_len | ETT-h1 | ETT-h1 | ETT-h2 | ETT-h2 | ETT-m1 | ETT-m1 | ETT-m2 | ETT-m2 |
| ------------- | :------: | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  |          | MSE    | MAE    | MSE    | MAE    | MSE    | MAE    | MSE    | MAE    |
| MeanForecast  |    96    |0.5325|0.5187|0.6679|0.5553|0.5941|0.5589|0.4453|0.4812|
| **LinearRegress** |    **96**    |**0.5312**|**0.5182**|**0.6647**|**0.5546**|<u>**0.5918**</u>|**0.5582**|**0.4451**|**0.4815**|
| MeanForecast  |   192    |0.6028|0.5585|0.8093|0.6248|0.6361|0.5876|0.5572|0.5403|
| **LinearRegress** |  **192** |**0.6012**|**0.5578**|**0.8047**|**0.6236**|**<u>0.6338</u>**|**0.587**|**0.5562**|**0.5401**|
| MeanForecast  |   336    |0.6582|0.585|0.9048|0.6768|0.714|0.6325|0.7403|0.6218|
| **LinearRegress** |  **336** |**0.6563**|**0.5841**|**0.8997**|**0.6753**|<u>**0.7114**</u>|**0.6321**|**0.7373**|**0.6206**|
| MeanForecast  |   720    |0.6724|0.5988|0.9155|0.6984|0.8098|0.6906|0.8542|0.6906|
| **LinearRegress** |  **720** |**0.6706**|**0.5979**|**0.9115**|**0.697**|<u>**0.807**</u>|**0.6902**|**0.8477**|**0.6882**|

