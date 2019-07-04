# 卷积神经网络 AlexNet

<a name="XaiKR"></a>
### 1.介绍
LeNet 是最早推动深度学习领域发展的卷积神经网络之一。这项由 Yann LeCun 完成的开创性工作自 1988 年以来多次成功迭代之后被命名为 LeNet5。AlexNet 是 Alex Krizhevsky 等人在 2012 年发表的《ImageNet Classification with Deep Convolutional Neural Networks》论文中提出的，并夺得了 2012 年 ImageNet LSVRC 的冠军，引起了很大的轰动。AlexNet 可以说是具有历史意义的一个网络结构，在此之前，深度学习已经沉寂了很长时间，自 2012 年 AlexNet 诞生之后，后面的 ImageNet 冠军都是用卷积神经网络（CNN）来做的，并且层次越来越深，使得CNN成为在图像识别分类的核心算法模型，带来了深度学习的大爆发。本文将详细讲解 AlexNet 模型及其使用 Keras 实现过程。开始之前，先介绍一下卷积神经网络。<br />
<a name="zB8wT"></a>
### 2. 卷积神经网络
<a name="pSpej"></a>
#### 2.1 卷积层
卷积是一种数学运算，它采用某种方式将一个函数“应用”到另一个函数，结果可以理解为两个函数的“混合体”。不过，这对检测图像中的目标有何帮助？事实证明，卷积非常擅长检测图像中的简单结构，然后结合这些简单特征来构造更复杂的特征。在卷积网络中，会在一系列的层上发生此过程，每层对前一层的输出执行一次卷积。卷积运算的目的是**提取输入的不同特征**，第一层卷积层可能只能提取一些低级的特征如边缘、线条和角等层级，更多层的网路能从低级特征中迭代提取更复杂的特征。<br />那么，您会在计算机视觉中使用哪种卷积呢？要理解这一点，首先了解图像到底是什么。图像是一种二阶或三阶字节数组，二阶数组包含宽度和高度 2 个维度，三阶数组有 3 个维度，包括宽度、高度和通道，所以灰阶图是二阶的，而 RGB 图是三阶的（包含 3 个通道）。字节的值被简单解释为整数值，描述了必须在相应像素上使用的特定通道数量。所以基本上讲，在处理计算机视觉时，可以将一个图像想象为一个 2D 数字数组（对于 RGB 或 RGBA 图像，可以将它们想象为 3 个或 4 个 2D 数字数组的相互重叠）。<br />![微信图片_20190619185025.png](https://cdn.nlark.com/yuque/0/2019/png/381685/1560941492137-2d570121-c014-4640-b59c-15344af9da2f.png#align=left&display=inline&height=270&name=%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190619185025.png&originHeight=812&originWidth=1192&size=830329&status=done&width=397)<br />图1：卷积运算示意图（左输入，中过滤器，右输出）[点我看计算演示](https://cs231n.github.io/assets/conv-demo/index.html)<br />应该注意的是，步幅和过滤器大小是超参数，这意味着模型不会学习它们。所以您必须应用科学思维来确定这些数量中的哪些值最适合您的模型。对于卷积，您需要理解的最后一个概念是填充。如果您的图像无法在整数次内与过滤器拟合（将步幅考虑在内），那么您必须填充图像。可通过两种方式实现此操作：VALID 填充和 SAME 填充。基本上讲，VALID 填充丢弃了图像边缘的所有剩余值。也就是说，如果过滤器为 2 x 2，步幅为 2，图像的宽度为 3，那么 VALID 填充会忽略来自图像的第三列值。SAME 填充向图像边缘添加值（通常为 0）来增加它的维数，直到过滤器能够拟合整数次。这种填充通常以对称方式进行的（也就是说，会尝试在图像的每一边添加相同数量的列/行）。
<a name="Cyd8w"></a>
#### 2.2 激活层
激活层主要是激活函数的作用，那么什么是激活函数呢？在神经网络中，当输入激励达到一定强度，神经元就会被激活，产生输出信号。模拟这一细胞激活过程的函数，就叫激活函数。将神经元的输出 f，作为其输入 x 的函数，对其建模的标准方法是用![](https://cdn.nlark.com/yuque/0/2019/png/381685/1560913957055-6a957c44-6f57-4489-a676-9486c9036929.png#align=left&display=inline&height=23&originHeight=23&originWidth=98&size=0&status=done&width=98)或者 sigmoid 函数![](https://cdn.nlark.com/yuque/0/2019/png/381685/1560913957082-d27f4e36-d9ba-4ce4-be7d-b6eb89d24baf.png#align=left&display=inline&height=29&originHeight=29&originWidth=108&size=0&status=done&width=108)。就梯度下降的训练时间而言，AlexNet 提出了比上面方式快 6 倍的 ReLu 函数![](https://cdn.nlark.com/yuque/0/2019/png/381685/1560914792065-bcf6c3b0-e2da-4574-a3f6-f887815f01f7.png#align=left&display=inline&height=23&originHeight=23&originWidth=109&size=0&status=done&width=109)。ReLU 全称为修正线性单元（Rectified Linear Units）是一种针对元素的操作（应用于每个像素），并将特征映射中的所有负像素值替换为零的非线性操作。其目的是在卷积神经网络中引入非线性因素，因为在实际生活中我们想要用神经网络学习的数据大多数都是非线性的（卷积是一个线性运算 —— 按元素进行矩阵乘法和加法，所以我们希望通过引入 ReLU 这样的非线性函数来解决非线性问题）。<br />![max.png](https://cdn.nlark.com/yuque/0/2019/png/381685/1560946353287-043c2c23-ef2a-4549-9c32-ddb8fe2e3f14.png#align=left&display=inline&height=186&name=max.png&originHeight=280&originWidth=372&size=25775&status=done&width=247)<br />图2: ReLU 函数（输入小于0则输出为0，输入大于0则输出原值）
<a name="fdfji"></a>
#### 2.3 池化层
您会在卷积网络中看到的另一种重要的层是池化层。池化层具有多种形式：最大值，平均值，求和等。但最常用的是最大池化，其中输入矩阵被拆分为相同大小的分段，使用每个分段中的最大值来填充输出矩阵的相应元素。池化层可以被认为是由间隔为 s 个像素的池单元网格组成，每个池汇总了以池单元的位置为中心的大小为 z×z 的邻域。如果我们设置 s = z（池化窗口大小与步长相同），我们获得在 CNN 中常用的传统的局部合并。 如果我们设置 s<z（每次移动的步长小于池化的窗口长度），我们就获得重叠池化。在 AlexNet 中首次使用重叠池化来避免过拟合。<br />![pooling.png](https://cdn.nlark.com/yuque/0/2019/png/381685/1560946642128-7d71280a-35cc-49fc-9805-4c5ee658cc7f.png#align=left&display=inline&height=179&name=pooling.png&originHeight=179&originWidth=301&size=12034&status=done&width=301)<br />图3：最大池化（左 16X16 大小分成了 4 块，黑子圈中是最大的数字）<br />

<a name="lgcOS"></a>
#### 2.4 全连接层
全连接层是一个传统的多层感知器，它在输出层使用 softmax 激活函数（也可以使用其他分类器，比如 SVM）。“完全连接”这个术语意味着前一层中的每个神经元都连接到下一层的每个神经元。 这是一种普通的卷积网络层，其中前一层的所有输出被连接到下一层上的所有节点。卷积层转换为全连接层时，总神经元个数不变。
<a name="RuOKr"></a>
### 3. AlexNet 模型
<a name="3qrnE"></a>
#### 3.1 模型介绍
![CNN.png](https://cdn.nlark.com/yuque/0/2019/png/381685/1560944024063-733199c6-d1c6-4df1-b680-a914571b6799.png#align=left&display=inline&height=375&name=CNN.png&originHeight=375&originWidth=1216&size=105070&status=done&width=1216)<br />图4：AlexNet 模型（ 5 卷积层+ 3 全连接层共 8 层神经网络，使用 2GPU 故分上下两部分）<br />AlexNet 模型包含 6 千万个参数和 65 万个神经元，包含 5 个卷积层，其中有几层后面跟着最大池化（max-pooling）层，以及 3 个全连接层，最后还有一个 1000 路的 softmax 层。为了加快训练速度，AlexNet 使用了 Relu 非线性激活函数以及一种高效的基于 GPU 的卷积运算方法。为了减少全连接层的过拟合，AlexNet 采用了最新的 “Dropout”防止过拟合方法，该方法被证明非常有效。
<a name="xNyJE"></a>
#### 3.2 **局部归一化（Local Response Normalization，简称LRN）**
在神经生物学有一个概念叫做“侧抑制”（lateral inhibitio），指的是被激活的神经元抑制相邻神经元。归一化（normalization）的目的是“抑制”，局部归一化就是借鉴了“侧抑制”的思想来实现局部抑制，尤其当使用 ReLU 时这种“侧抑制”很管用，因为 ReLU 的响应结果是无界的（可以非常大），所以需要归一化。使用局部归一化的方法有助于增加泛化能力。<br />![image.png](https://cdn.nlark.com/yuque/0/2019/png/381685/1560948416974-103a0395-a099-402f-bc32-06ff6c5031be.png#align=left&display=inline&height=108&name=image.png&originHeight=108&originWidth=534&size=25878&status=done&width=534)
<a name="DbRZd"></a>
### 4. AlexNet 过拟合处理
<a name="1vjva"></a>
#### 4.1 数据扩充
减少图像数据过拟合最简单最常用的方法，是使用标签-保留转换，人为地扩大数据集。AlexNet 模型使用两种不同的形式，这两种形式都允许转换图像用很少的计算量从原始图像中产生，所以转换图像不需要存储在磁盘上。数据扩充的第一种形式由生成图像转化和水平反射组成。数据扩充的第二种形式包含改变训练图像中 RGB 通道的强度。<br />![LRN.png](https://cdn.nlark.com/yuque/0/2019/png/381685/1560948611624-ac0d3eef-e816-403e-ae98-0959f733ffa7.png#align=left&display=inline&height=623&name=LRN.png&originHeight=829&originWidth=442&size=357278&status=done&width=332)<br />图5：数据扩充三种方式
<a name="Zw3XX"></a>
#### 4.2 Dropout
对某一层神经元，Dropout 做的就是以 0.5 的概率将每个隐层神经元的输出设置为零。以这种方式 “Dropped out” 的神经元既不用于前向传播，也不参与反向传播。所以每次提出一个输入，该神经网络就尝试一个不同的结构，所有这些结构之间共享权重。因为神经元不能依赖于其他特定神经元而存在，所以这种技术降低了神经元复杂的互适应关系。正因如此，要被迫学习更为鲁棒的特征，这些特征在结合其他神经元的一些不同随机子集时有用。如果没有 Dropout，AlexNet 网络会表现出大量的过拟合。<br />![Dropout.png](https://cdn.nlark.com/yuque/0/2019/png/381685/1560948915869-eb1d0a6b-eb48-4238-9ef6-407f22620514.png#align=left&display=inline&height=280&name=Dropout.png&originHeight=373&originWidth=556&size=53964&status=done&width=417)<br />图6：Dropout示意图
<a name="6CISN"></a>
### 5. 源码解读
<a name="3o4kz"></a>
#### 5.1数据集和导入依赖库
```python
# (1) Importing dependency
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
np.random.seed(1000)
# (2) Get Data
import tflearn.datasets.oxflower17 as oxflower17
x, y = oxflower17.load_data(one_hot=True)
# (3) Create a sequential model
model = Sequential()
```
AlexNet 模型建立在千分类问题上，其算力对计算机要求很高。这里我们为了简单复现，使用了 TensorFlow 的数据集 oxflower17 ，此数据集对花朵进行17 分类，每个分类有 80 张照片。Keras 包含许多常用神经网络构建块的实现，例如层、目标、激活函数、优化器和一系列工具，可以更轻松地处理图像和文本数据。在 Keras 中有两类主要的模型：Sequential 顺序模型和使用函数式 API 的 Model 类模型。这里使用 Sequential 模型。
<a name="BlYfX"></a>
#### 5.3 **第一次卷积+池化**
```python
# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())
```
![image.png](https://cdn.nlark.com/yuque/0/2019/png/381685/1560942750102-8e818667-ab3b-4c5e-b394-fe02963f8f45.png#align=left&display=inline&height=434&name=image.png&originHeight=434&originWidth=511&size=54073&status=done&width=511)<br />**卷积层 1 大小 224X224X3，卷积核大小 11X11X3，数量 48，步长为 4。**<br />关于卷积层的计算如下：

- 输入数据体尺寸为![](https://cdn.nlark.com/yuque/__latex/3c113d427b7607d159d1bbf85856bcc1.svg#card=math&code=W_1%2AH_1%2AD_1&height=24&width=95)
- 4个超参数（模型不会学习优化）：
  1. 滤波器数量![](https://cdn.nlark.com/yuque/__latex/a5f3c6a11b03839d46af9fb43c97c188.svg#card=math&code=K&height=24&width=14)
  1. 滤波器空间尺寸![](https://cdn.nlark.com/yuque/__latex/800618943025315f869e4e1f09471012.svg#card=math&code=F&height=24&width=12)
  1. 卷积运算步长![](https://cdn.nlark.com/yuque/__latex/5dbc98dcc983a70728bd082d1a47546e.svg#card=math&code=S&height=24&width=10)
  1. 零填充数量（SAME 填充）![](https://cdn.nlark.com/yuque/__latex/44c29edb103a2872f519ad0c9a0fdaaa.svg#card=math&code=P&height=24&width=12)
- 输出数据体尺寸为![](https://cdn.nlark.com/yuque/__latex/67b123d02899c2973ffeb2f7d30acd99.svg#card=math&code=W_2%2AH_2%2AD_2&height=24&width=95)
  - ![](https://cdn.nlark.com/yuque/__latex/d7b4fefd0abdcab3f76d59cc333c898a.svg#card=math&code=W_2%20%3D%20%5Cfrac%7B%28W_1-F%2B2P%29%7D%7BS%7D%2B1&height=41&width=186)
  - ![](https://cdn.nlark.com/yuque/__latex/d39a6a005991434ff40c29bc39729c15.svg#card=math&code=H_2%3D%5Cfrac%7B%28H_1-F%2B2P%29%7D%7BS%7D%2B1&height=41&width=182)
  - ![](https://cdn.nlark.com/yuque/__latex/ce5d82eb8dd9f2869ea2743b0071d6c8.svg#card=math&code=D_2%3DK&height=24&width=57)

这里 W1=224，H1=224，D1=3，K=48，F=11，S=4，P=1.5。

计算卷积层 2 有 W2=（224-11+3）/4+1=55，同理 H2=55，D2=K*2=96。

经过卷积运算后，输出特征图像大小为 55X55X96。<br />这里使用了最大池化，步长为 S=2，则 W=（55-3）/2+1=27。<br />再经过池化后，输出特征图像大小为 27X27X96。
<a name="nZdHw"></a>
#### 5.4 第二次卷积+池化
```python
# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())
```
![image.png](https://cdn.nlark.com/yuque/0/2019/png/381685/1560942797805-65e0bf5d-7da5-4f79-93f3-6cc6c8ee25c8.png#align=left&display=inline&height=430&name=image.png&originHeight=430&originWidth=655&size=67793&status=done&width=655)<br />**卷积层 2 大小 55X55X96，卷积核大小 5X5，数量为 128 个，步长为 1。**

同理，可以计算卷积后得到特征图像大小为 27X27X256。<br />经过池化，输出特征图像大小为 13X13X256。**
<a name="4W5YY"></a>
#### 5.5 第三次卷积
```python
# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())
```
![image.png](https://cdn.nlark.com/yuque/0/2019/png/381685/1560942004332-5245e1be-cd52-46e8-a318-efbdf3d55c38.png#align=left&display=inline&height=405&name=image.png&originHeight=405&originWidth=449&size=45975&status=done&width=449)<br />经过卷积后，特征图像大小为 13X13X384。<br />
<a name="Vc4xI"></a>
#### 5.6 第四次卷积
```python
# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())
```
![image.png](https://cdn.nlark.com/yuque/0/2019/png/381685/1560941980746-fa29b258-970d-4b98-9b3d-f317d1f45d69.png#align=left&display=inline&height=342&name=image.png&originHeight=342&originWidth=438&size=34165&status=done&width=438)<br />经过卷积后，特征图像大小为 13X13X284。<br />
<a name="hmYx3"></a>
#### 5.7 第五次卷积+池化
```python
# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())
```
![image.png](https://cdn.nlark.com/yuque/0/2019/png/381685/1560942854961-9b96e01c-2e2f-4162-8478-d0f7ed868777.png#align=left&display=inline&height=384&name=image.png&originHeight=384&originWidth=482&size=37392&status=done&width=482)<br />经过卷积后，特征图像大小为 13X13X256。

经过池化后，特征图像大小为 6X6X256。
<a name="XnKEA"></a>
#### 5.8 全连接层6
```python
# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())
```
![image.png](https://cdn.nlark.com/yuque/0/2019/png/381685/1560941868658-b0122117-45f9-4447-8c17-940e454e9612.png#align=left&display=inline&height=378&name=image.png&originHeight=378&originWidth=262&size=23862&status=done&width=262)<br />全连接层6大小为 6X6X256，共 4096 个神经元，输出 4096X1 的向量。<br />
<a name="TtIvm"></a>
#### 5.9 全连接层7
```python
# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())
```
![image.png](https://cdn.nlark.com/yuque/0/2019/png/381685/1560941845971-adbdd627-10e8-4238-8e69-174352dc82fe.png#align=left&display=inline&height=364&name=image.png&originHeight=364&originWidth=158&size=15338&status=done&width=158)<br />全连接层7大小为 4096X1，共 4096 个神经元，输出 4096X1 的向量。<br />
<a name="3XSuG"></a>
#### 5.10 全连接层8
```python
# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())
```
![image.png](https://cdn.nlark.com/yuque/0/2019/png/381685/1560941783083-0316a3a9-2aa7-4006-a1cb-d989921fc0a2.png#align=left&display=inline&height=364&name=image.png&originHeight=364&originWidth=143&size=11319&status=done&width=143)<br />全连接层8输入大小为4096X1，共 4096 个神经元，输出 1000X1 的向量。<br />
<a name="aBxDX"></a>
#### 5.11 输出层及训练
```python
# Output Layer
model.add(Dense(17))
model.add(Activation('softmax'))
model.summary()
# (4) Compile 
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
# (5) Train
model.fit(x, y, batch_size=64, epochs=1, verbose=1, validation_split=0.2, shuffle=True)
```
最后在全连接层经过softmax激活函数后得到结果。<br />![image.png](https://cdn.nlark.com/yuque/0/2019/png/381685/1560943124263-76a4fff3-2328-4303-804c-9628484b5ab0.png#align=left&display=inline&height=361&name=image.png&originHeight=361&originWidth=612&size=35304&status=done&width=612)<br />在结果损失函数图中，训练集的损失值在迭代23次时达到最小。由于数据量有限，测试集的损失值无法降到理想位置。（横坐标是迭代次数，纵坐标是损失函数的值）<br />![image.png](https://cdn.nlark.com/yuque/0/2019/png/381685/1560943130608-4db69a4d-9a39-4708-bb79-162564ac69c5.png#align=left&display=inline&height=345&name=image.png&originHeight=345&originWidth=587&size=35640&status=done&width=587)<br />可以看出，在训练集上可以达到近90%的准确率。（横坐标是迭代次数，纵坐标是准确率）<br />由于数据集太小，测试集的准确率无法达到理想的值。
<a name="J01sk"></a>
### 6. 总结与展望
目前，您可以在  [Mo ](http://www.momodel.cn:8899/)平台中找到基于 AlexNet 的项目 [Flower](http://www.momodel.cn:8899/explore/5cff0ee61afd941c7e304adb?type=app)，此项目对原文的千分类进行整合，最终做成花卉的17分类。您在学习的过程中，遇到困难或者发现我们的错误，可以随时联系我们。<br />项目源码地址：[http://www.momodel.cn:8899/explore/5cff0ee61afd941c7e304adb?type=app](http://www.momodel.cn:8899/explore/5cff0ee61afd941c7e304adb?type=app)

总结一下 AlexNet 的主要贡献：

1. 2 路 GPU 实现，加快了训练速度
1. Relu 非线性激活函数，减少训练时间，加快训练速度
1. 重叠池化，提高精度，不容易产生过拟合
1. 为了减少过拟合，使用了数据扩充和 “Dropout”
1. 使用局部响应归一化，提高精度
1. 5 个卷积层+ 3 个全连接层，结构性能良好
<a name="WOklj"></a>
### 7. 参考

- 论文：[http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) 
- 论文翻译版：[https://zhuanlan.zhihu.com/p/35400048](https://zhuanlan.zhihu.com/p/35400048)
- 数据集：[http://www.robots.ox.ac.uk/~vgg/data/flowers/17/](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
- Keras：[https://keras.io/zh/](https://keras.io/zh/)
- 算法：[https://www.mydatahack.com/building-alexnet-with-keras/](https://www.mydatahack.com/building-alexnet-with-keras/)
- 博客：[https://my.oschina.net/u/876354/blog/1633143](https://my.oschina.net/u/876354/blog/1633143)
- 博客：[https://www.ibm.com/developerworks/cn/cognitive/library/cc-convolutional-neural-network-vision-recognition/index.html](https://www.ibm.com/developerworks/cn/cognitive/library/cc-convolutional-neural-network-vision-recognition/index.html)

<a name="oxtTX"></a>
### 关于我们
**Mo**（网址：[**momodel.cn**](http://www.momodel.cn:8899/)）是一个支持 Python 的**人工智能在线建模平台**，能帮助你快速开发、训练并部署模型。

---

**Mo 人工智能俱乐部** 是由网站的研发与产品设计团队发起、致力于降低人工智能开发与使用门槛的俱乐部。团队具备大数据处理分析、可视化与数据建模经验，已承担多领域智能项目，具备从底层到前端的全线设计开发能力。主要研究方向为大数据管理分析与人工智能技术，并以此来促进数据驱动的科学研究。

目前俱乐部每周六在杭州举办以机器学习为主题的线下技术沙龙活动，不定期进行论文分享与学术交流。希望能汇聚来自各行各业对人工智能感兴趣的朋友，不断交流共同成长，推动人工智能民主化、应用普及化。<br />![image.png](https://cdn.nlark.com/yuque/0/2019/png/307794/1560565564936-bcd9ec1e-8e47-4373-ba0d-1ab3a696aee4.png#align=left&display=inline&height=175&name=image.png&originHeight=349&originWidth=720&size=170790&status=done&width=360)



