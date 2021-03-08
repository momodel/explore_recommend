# 家庭用户的用电预测——在 Keras 深度学习库中开发多变量时间序列预测的 LSTM 模型

<a name="THcbz"></a>

用电量可以反映一个国家经济发展的水平，对用电量进行全面的理解有助于减少家庭的电费支出。<br />对企业而言，对用电量全面的理解有助于提高经营的效率。对于政府而言，全面的了解用电量可以减少政府对发电，供电等需要的基建投资，为政府对当地经济发展制定更好更全面的规划。

鉴于智能电表的兴起以及太阳能电池板等发电技术的广泛采用，有大量的用电数据可供选择。该数据代表了功率的相关变量，这些变量又可用于建模甚至预测未来的电力消耗。像长期短期记忆网络（LSTM）这样的神经网络能够处理多个输入变量的问题。这在时间序列预测中具有很大的益处，而传统的线性方法难以适应多变量或多输入预测问题。

<a name="7813af34"></a>
### 第一步 准备工作

在本教程中，您将了解如何在 Keras 深度学习库中开发多变量时间序列预测的 LSTM 模型。<br />完成本教程后，您将知道：

- 如何将原始数据集转换为可用于时间序列预测的类型
- 如何搭建解决多变量时间序列预测问题的 LSTM 模型
- 如何做出预测并将结果重新调整到原始单位

**特征介绍**<br />**

1. date: 日期格式为 dd/mm/yy
1. time: 时间格式为 hh:mm:ss
1. Global_active_power: 家庭消耗的总有功功率（千瓦），在交流电路中，电源在一个周期内发出瞬时功率的平均值(或负载电阻所消耗的功率)，称为"有功功率"
1. Global_reactive_power: 家庭消耗的总无功功率（千瓦），在具有电感或电容的电路中，在每半个周期内，把电源能量变成磁场(或电场)能量贮存起来，然后，再释放，又把贮存的磁场（或电场）能量再返回给电源，只是进行这种能量的交换，并没有真正消耗能量，我们把这个交换的功率值，称为" 无功功率"
1. voltage: 平均电压（伏特）
1. Global_intensity: 平均电流强度（安培)
1. sub_metering_1: 厨房的有功功率
1. sub_metering_2: 用于洗衣机等电器的有功功率
1. sub_metering_3: 空调热水器等电器的有功功率

这里我们使用的是时间序列预测模型，利用历史数据来预测之后的 Global_active_power。

<a name="0d2a88fe"></a>
### 第二步 导入数据并进行数据预处理

```python
# 将txt文档读入并转换为 csv 文件格式
df = pd.read_csv(path, sep=';', 
                 parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True, 
                 low_memory=False, na_values=['nan','?'], index_col='dt')
```

上面代码执行了以下操作：<br />1）将 'Date' 和 'Time' 两列合并为 'dt'<br />2）将上面的数据转换为时间序列类型，将时间作为索引。

```python
# 我们查看前 5 条数据
df.head()
```
![屏幕快照 2019-05-17 下午1.26.02.png](https://cdn.nlark.com/yuque/0/2019/png/349862/1558070784379-391dd7f8-0cfa-4e78-8e31-05ab1b424da0.png#align=left&display=inline&height=224&name=%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-05-17%20%E4%B8%8B%E5%8D%881.26.02.png&originHeight=224&originWidth=1005&size=42357&status=done&width=1005)<br />我们可以看出 Global_active_power 大于 Global_reactive_power， voltage 基本稳定在 233 伏特。

```python
# 了解数据的分布
df.describe()
```

![屏幕快照 2019-05-17 下午1.38.15.png](https://cdn.nlark.com/yuque/0/2019/png/349862/1558071521599-1ae72546-e686-4889-b25c-31f9ea4f2a2e.png#align=left&display=inline&height=155&name=%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-05-17%20%E4%B8%8B%E5%8D%881.38.15.png&originHeight=195&originWidth=938&size=45248&status=done&width=746)<br />我们可以通过上表了解数据的分布特征，比如均值和方差，还有最小值等等。

<a name="gxH0n"></a>
#### 处理缺失值
在原始计量数据，特别是用户电量抽取过程中，发现存在缺失现象。若将这些值抛弃掉，会严重影响用电预测的结果。为了达到较好的建模效果，需要对缺失值进行处理。

```python
# 找到所有有缺失值的列
total = df.isnull().sum().sort_values(ascending=False)
display(total)
# 用各列的均值填充缺失值
for j in range(0,7):        
        df.iloc[:,j] = df.iloc[:,j].fillna(df.iloc[:,j].mean())
# 查看是否还有缺失值
df.isnull().sum()
```
<a name="d35b5686"></a>
### 
<a name="w1eFy"></a>
### 第三步 数据可视化

大部分真实的数据集都难以观察，因为它们有很多列变量，以及很多行数据。理解信息这方面大量都依赖于视觉。查看数据基本等价于了解数据。然而，基本上我们只能理解视觉上的二维或者三维数据，最好是二维。所以数据可视化能够帮助我们提高对数据的理解。

对数据集中呈现的结构和相关性进行观察，会让它们易于理解。一个准确的机器学习模型给出的预测，应当能够反映出数据集中所体现的结构和相关性。要明确一个模型给出的预测是否可信，对这些结构和相关性进行理解是首当其冲的。

<a name="06e0d2f9"></a>
#### 了解数据分布
我们可以使用  resample 函数使特征按不同单位进行聚合。例如:使用参数 'H' 调用此函数使时间索引的数据按小时聚合。<br />下面我们对 Global_active_power 按天进行聚合，并比较它的总和和平均值。 可以看出，重采样数据集的平均值和总和具有相似的结构。

```python
# 对 Global_active_power 数据按天进行聚合，并比较均值和总和
df.Global_active_power.resample('D').sum().plot(title='Global_active_power resampled over day for sum') 
plt.tight_layout()
plt.show()   

df.Global_active_power.resample('D').mean().plot(title='Global_active_power resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()
```

![output_21_0.png](https://cdn.nlark.com/yuque/0/2019/png/349862/1558071863420-d09515d0-10d4-4982-b15f-d88ad8aa9f65.png#align=left&display=inline&height=244&name=output_21_0.png&originHeight=280&originWidth=424&size=24154&status=done&width=370)![output_21_1.png](https://cdn.nlark.com/yuque/0/2019/png/349862/1558071879582-9974e984-d82e-4411-8f33-c776f28bbd73.png#align=left&display=inline&height=245&name=output_21_1.png&originHeight=280&originWidth=424&size=19753&status=done&width=371)

```python
# 对 'Global_active_power' 按季度进行聚合
df['Global_active_power'].resample('Q').mean().plot(kind='bar')
plt.xticks(rotation=60)
plt.ylabel('Global_active_power')
plt.title('Global_active_power per quarter (averaged over quarter)')
plt.show()

# 对'Voltage' 按月进行聚合
df['Voltage'].resample('M').mean().plot(kind='bar', color='red')
plt.xticks(rotation=60)
plt.ylabel('Voltage')
plt.title('Voltage per quarter (summed over quarter)')
plt.show()
```

![output_23_0.png](https://cdn.nlark.com/yuque/0/2019/png/349862/1558071963690-214839e9-1dbb-411d-af75-82720eace4f1.png#align=left&display=inline&height=296&name=output_23_0.png&originHeight=364&originWidth=414&size=19823&status=done&width=337)![output_24_0.png](https://cdn.nlark.com/yuque/0/2019/png/349862/1558072023859-69089f83-b6e0-498a-bc58-d9651d530c5f.png#align=left&display=inline&height=309&name=output_24_0.png&originHeight=364&originWidth=416&size=22158&status=done&width=353)<br />我们可以看出每个月电压平均值变化幅度非常小，基本保持稳定。

```python
# 下面我们比较对不同特征以天进行重采样的数值
cols = [0, 1, 2, 3, 5, 6]
i = 1
groups=cols
# 统计以天进行重采样的平均值
values = df.resample('D').mean().values
# 对每个column进行绘图
plt.figure(figsize=(15, 10))
for group in groups:
    # 对每个特征添加子图
	plt.subplot(len(cols), 1, i)
    # 进行绘图
	plt.plot(values[:, group])
    # 添加标题
	plt.title(df.columns[group], y=0.75, loc='right')
    # 更换子图位置
	i += 1
plt.show()
```


![output_26_0.png](https://cdn.nlark.com/yuque/0/2019/png/349862/1558072112835-9c492248-e516-44df-af18-5b67c0cd834e.png#align=left&display=inline&height=578&name=output_26_0.png&originHeight=578&originWidth=880&size=166191&status=done&width=880)

```python
# 下面我们看看 ‘Global_active_power‘ 数值分布情况
sns.distplot(df['Global_active_power']);
```

![output_27_0.png](https://cdn.nlark.com/yuque/0/2019/png/349862/1558072146557-3bf39d5e-6e10-49c9-8595-d84f7421a1da.png#align=left&display=inline&height=268&name=output_27_0.png&originHeight=268&originWidth=375&size=10178&status=done&width=375)

可以看出家庭消耗的总有功功率主要集中在 0-2kw 范围内

<a name="l9dSL"></a>
#### 特征相关性分析

```python
# 查看 'Global_intensity' 和 'Global_active_power' 特征之间的关系
data_returns = df.pct_change()
sns.jointplot(x='Global_intensity', y='Global_active_power', data=data_returns)  
plt.show()
```

![output_30_0.png](https://cdn.nlark.com/yuque/0/2019/png/349862/1558072232500-6850370f-d0e8-4042-8d02-e7b400246993.png#align=left&display=inline&height=375&name=output_30_0.png&originHeight=424&originWidth=421&size=11776&status=done&width=372)

```python
# 查看 'Voltage' 和 'Global_active_power' 之间的关系
sns.jointplot(x='Voltage', y='Global_active_power', data=data_returns)  
plt.show()
```


![output_31_0.png](https://cdn.nlark.com/yuque/0/2019/png/349862/1558072282219-37a56e44-4f68-4393-8a33-9f36e691203f.png#align=left&display=inline&height=375&name=output_31_0.png&originHeight=424&originWidth=421&size=11976&status=done&width=372)

从上面的两个图中可以看出 'Global_intensity' 和 'Global_active_power' 是线性相关的。 但 'Voltage' 和 'Global_active_power' 的相关性较低， 这是机器学习所要观察的。

```python
# 对各特征按月进行聚合
plt.title('resampled over month',size=15)
sns.heatmap(df.resample('M').mean().corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=plt.cm.RdBu, linecolor='white', annot=True)

# 对各特征按年进行聚合
plt.title('resampled over year',size=15)
sns.heatmap(df.resample('A').mean().corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=plt.cm.RdBu, linecolor='white', annot=True)
```
![屏幕快照 2019-05-17 下午1.55.12.png](https://cdn.nlark.com/yuque/0/2019/png/349862/1558072545607-463642f7-0968-464e-b55e-4f316deb8c71.png#align=left&display=inline&height=310&name=%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-05-17%20%E4%B8%8B%E5%8D%881.55.12.png&originHeight=372&originWidth=421&size=50483&status=done&width=351)![屏幕快照 2019-05-17 下午1.55.58.png](https://cdn.nlark.com/yuque/0/2019/png/349862/1558072584449-c31068ba-cd76-4314-a68a-eff004a854ed.png#align=left&display=inline&height=315&name=%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-05-17%20%E4%B8%8B%E5%8D%881.55.58.png&originHeight=373&originWidth=421&size=50871&status=done&width=355)<br />从上面可以看出，采用重采样技术可以改变特征之间的相关性， 这对于特征工程非常重要。
<a name="t0wYf"></a>
### [](http://www.momodel.cn:8899/workspace/5cde0ed11afd94371e5697ff/app#%E7%AC%AC%E4%B8%89%E6%AD%A5-%E5%A4%9A%E5%8F%98%E9%87%8FLSTM%E9%A2%84%E6%B5%8B%E6%A8%A1%E5%9E%8B)
<a name="d1d0c098"></a>
### 第四步 多变量LSTM预测模型
在可以使用深度学习之前，必须将时间序列预测问题调整为监督学习问题，形成输入和输出序列对，利用前一时间的 Global_active_power 和其他特征预测当前时间的 Global_active_power。

```python
# 下面我们对 ‘Global_active_power' 向前移动一个单位
df['Global_active_power'].resample('h').mean().shift(1)
# 下面我们对 ‘Global_active_power' 向后移动一个单位
df['Global_active_power'].resample('h').mean().shift(-1)
```

因为这里我们预测 ‘Global_active_power' 不仅用到过去时间的 ‘Global_active_power' 还会用到其他的特征，比如：'Voltage‘，这时候我们把此类问题叫做多变量时间序列预测。下面我们展示将时间序列问题转换为监督学习问题的过程。

```python
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	dff = pd.DataFrame(data)
	cols, names = list(), list()
	# 输入序列(t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(dff.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# 预测序列 (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(dff.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# 组合起来
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# 丢掉NaN
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
```

我们将数据以小时进行聚合，这样也可以减少计算时间，可以快速获得测试模型的结果。 我们以小时进行聚合（原始数据以分钟为单位）。这将把数据大小从 2075259 减少到 34589，但依然保持数据的整体结构。

```python
# 将数据按小时聚合
df_resample = df.resample('h').mean() 
df_resample.shape
```


<a name="a00a8b7a"></a>
#### 对特征进行归一化

数据归一化处理是数据挖掘的一项基础工作。不同指标往往具有不同的量纲，数值间的差别可能很大，不进行处理可能会影响数据分析的结果。为了消除指标间的量纲和取值范围差异的影响，需要进行标准化处理，将数据按照比例进行缩放，使之落入特定的区域，以便于进行综合分析。

同样我们必须对用户电量的各指标进行处理，这里我们用到最小最大规范化。

```python
# 把所有特征进行规范化
values = df_resample.values 
# 特征归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# 转化为监督问题
```

```python
# 转化为监督问题
reframed = series_to_supervised(scaled, 1, 1)
# 删除不需要的特征
reframed.drop(reframed.columns[[8,9,10,11,12,13]], axis=1, inplace=True)
print(reframed.head())
```

<a name="5b2cd780"></a>
#### 把数据集拆分成训练集和测试集

这里，我们将前三年的数据作为训练集，后一年的样本作为测试集，并将数据改为 3 维格式。 

```python
# 对样本集拆分成训练集和测试集
values = reframed.values
n_train_time = 365*24*3
train = values[:n_train_time, :]
test = values[n_train_time:, :]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# 把数据转换为3维
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
```

<a name="u1wx6"></a>
#### 搭建 LSTM 模型
模型架构<br />1）LSTM 在第一个可见层中有 100 个神经元<br />2）丢弃 20％，防止过拟合<br />3）输出层中 1 个神经元，用于预测 Global_active_power<br />4）使用平均绝对误差（MAE）损失函数和随机梯度下降的 Adam 优化器<br />5）该模型 epoch 为 20，批次大小为 70 
<a name="nri3S"></a>
#### 
```python
# 搭建网络模型
model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# 训练网络
history = model.fit(train_X, train_y, epochs=20, batch_size=70, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# 统计 loss 值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
# 做出预测
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], 7))
```

![output_54_1.png](https://cdn.nlark.com/yuque/0/2019/png/349862/1558073265903-e17735fe-468a-42a8-bf3a-508094040f3d.png#align=left&display=inline&height=278&name=output_54_1.png&originHeight=278&originWidth=402&size=11734&status=done&width=402)<br />我们可以看出模型的收敛速度很快。

<a name="4ff81b8c"></a>
### 第五步 模型评估

预测模型对训练集进行预测而得到的准确率并不能很好地反映预测模型对未来的性能，为了有效判断一<br />个预测模型的性能表现，需要一组没有参加预测模型建立的数据集，并在该模型上评价预测模型的准确率，这组独立的数据集叫做测试集。在测试集进行预测并且评估，我们怎样对模型进行性能衡量？

回归问题的评价指标:通常用相对／绝对误差，平均绝对误差，均方误差，均方根误差等指标来衡量,分类问题的评价指标：准确率，精确率，召回率，ROC曲线，混淆矩阵。

我们将预测与测试数据集相结合，并反演缩放。<br />以预测值和实际值为原始尺度，我们可以计算模型的误差分数。 在这种情况下，我们计算出与变量本身相同的单位产生误差的均方根误差（RMSE)。

```python

#对预测值进行反演缩放
inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# 对真实值进行反演缩放
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# 计算 RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
```

```python
## 我们比较下 200 小时真实值和预测值 
aa=[x for x in range(200)]
plt.plot(aa, inv_y[:200], marker='.', label="actual")
plt.plot(aa, inv_yhat[:200], 'r', label="prediction")
plt.ylabel('Global_active_power', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show()
```

![output_60_0.png](https://cdn.nlark.com/yuque/0/2019/png/349862/1558073305799-cb697cb5-daad-4bad-8019-7e238f8d9c8a.png#align=left&display=inline&height=271&name=output_60_0.png&originHeight=271&originWidth=395&size=42994&status=done&width=395)

<a name="d6abd198"></a>
### 第六步 思考，如何进一步的改进模型？

能不能进一步的改进模型呢？下面提出了一些可能的改进模型的方案，有兴趣的话可以试一试哦。<br />1.在缺失值处理中利用其他的插值方法<br />2.使用复杂的模型

```
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  n_vars = 1 if type(data) is list else data.shape[1]
  dff = pd.DataFrame(data)
  cols, names = list(), list()
  # 输入序列(t-n, ... t-1)
  for i in range(n_in, 0, -1):
  cols.append(dff.shift(i))
  names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
  # 预测序列 (t, t+1, ... t+n)
  for i in range(0, n_out):
  cols.append(dff.shift(-i))
  if i == 0:
  names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
  else:
  names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # 组合起来
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # 丢掉NaN
  if dropnan:
  agg.dropna(inplace=True)
  return agg
```



3.调整 epoch 和 batch_size

<a name="25f9c7fa"></a>
### 第七步 总结

今天我们一起制作了一个家庭用户用电预测的应用，大家可以在项目源码地址 fork 这个项目<br />[http://www.momodel.cn:8899/explore/5cde0ed11afd94371e5697ff?type=app](http://www.momodel.cn:8899/explore/5cde0ed11afd94371e5697ff?type=app)

我们首先对数据进行预处理，处理缺失值；接着进行数据可视化，了解数据的结构和相关性；然后搭建 LSTM 模型，其中最为关键的是将问题转化为监督学习问题；最后我们对模型进行评估，并提出了优化模型的建议。

使用我们的模型同样也可以预测温湿度和股价等等，只需要略加修改就行，来做出自己的应用吧。

---

参考资料：<br />[https://wenku.baidu.com/view/3973baa6951ea76e58fafab069dc5022aaea46b9.html](https://wenku.baidu.com/view/3973baa6951ea76e58fafab069dc5022aaea46b9.html)<br />[https://blog.csdn.net/sinat_22510827/article/details/80996937](https://blog.csdn.net/sinat_22510827/article/details/80996937)<br />[https://blog.csdn.net/weixin_40651515/article/details/83895707](https://blog.csdn.net/weixin_40651515/article/details/83895707)<br />[https://www.jianshu.com/p/bebf8ca6a946](https://www.jianshu.com/p/bebf8ca6a946)

---

**<br />Mo**（网址：**[momodel.cn](http://link.zhihu.com/?target=http%3A//momodel.cn/)**）是一个支持 Python 的**人工智能在线建模平台**，能帮助你快速开发、训练并部署模型。

---


**Mo 人工智能俱乐部** 是由网站的研发与产品设计团队发起、致力于降低人工智能开发与使用门槛的俱乐部。团队具备大数据处理分析、可视化与数据建模经验，已承担多领域智能项目，具备从底层到前端的全线设计开发能力。主要研究方向为大数据管理分析与人工智能技术，并以此来促进数据驱动的科学研究。<br />目前俱乐部每周六在杭州举办以机器学习为主题的线下技术沙龙活动，不定期进行论文分享与学术交流。希望能汇聚来自各行各业对人工智能感兴趣的朋友，不断交流共同成长，推动人工智能民主化、应用普及化。



 <div>
    <h3>微信公众号: MomodelAI</h3>
    <img src='https://mo-imgs.momodel.cn/WeChatPublicAccount.png' height="220"/>
 </div>
 <br/>
 <br/>

<div>
    <h3>添加管理员微信加入AI俱乐部群聊</h3>
    <img src='https://mo-imgs.momodel.cn/wechat_office_contact.png' height="210">
</div>

