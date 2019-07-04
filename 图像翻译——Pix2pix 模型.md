# 图像翻译——Pix2pix 模型

<a name="36ee8972"></a>
### 1.介绍

图像处理、计算机图形学和计算机视觉中的许多问题都可以被视为将**输入图像“翻译”成相应的输出图像**。 “翻译”常用于语言之间的翻译，比如中文和英文的之间的翻译。但图像翻译的意思是**图像与图像之间以不同形式的转换**。比如：一个图像场景可以以RGB图像、梯度场、边缘映射、语义标签映射等形式呈现，其效果如下图。

![](https://cdn.nlark.com/yuque/0/2019/png/307794/1560565027879-1e8e7ff6-16f8-4503-8df1-edfe3d37faf4.png#align=left&display=inline&height=188&originHeight=585&originWidth=1555&size=0&status=done&width=500)

传统图像转换过程中都是针对具体问题采用特定算法去解决；而这些过程的本质都是根据**像素点（输入信息）对像素点做出预测(predict from pixels to pixels)**，Pix2pix的目标就是建立一个通用的架构去解决以上所有的图像翻译问题，使得我们不必要为每个功能都重新设计一个损失函数。

<a name="2ab5870d"></a>
### 2. 核心思想

<a name="f6da9350"></a>
#### **2.1 图像建模的结构化损失**

图像到图像的翻译问题通常是根据像素分类或回归来解决的。这些公式将输出空间视为**“非结构化”**，即在给定输入图像的情况下，每个输出像素被视为与所有其他像素有条件地独立。而cGANs（ conditional-GAN）的不同之处在于学习结构化损失，并且理论上可以惩罚输出和目标之间的任何可能结构。

<a name="0a5c99b7"></a>
#### 2.2 c**GAN**

在此之前，许多研究者使用 GAN 在修复、未来状态预测、用户约束引导的图像处理、风格迁移和超分辨率方面取得了令人瞩目的成果，但每种方法都是针对特定应用而定制的。Pix2pix框架不同之处在于没有特定应用。它在生成器和判别器的几种架构选择中也与先前的工作不同。对于生成器，我们使用基于“U-Net”的架构；对于鉴别器，我们使用卷积“PatchGAN”分类器，其仅在image patches（图片小块）的尺度上惩罚结构。

Pix2pix 是借鉴了 cGAN 的思想。cGAN 在输入 G 网络的时候不光会输入噪音，还会输入一个条件（condition），G 网络生成的 fake images 会受到具体的 condition 的影响。那么如果把一副图像作为 condition，则生成的 fake images  就与这个 condition images 有对应关系，从而实现了一个 Image-to-Image Translation  的过程。Pixpix 原理图如下：

![](https://cdn.nlark.com/yuque/0/2019/png/307794/1560504644099-9b642982-b978-41ff-8db8-a8bc39d4f23d.png#align=left&display=inline&height=182&originHeight=238&originWidth=653&size=0&status=done&width=500)

Pix2pix 的网络结构如上图所示，生成器 G 用到的是 U-Net 结构，输入的轮廓图 ![](https://cdn.nlark.com/yuque/0/2019/svg/307794/1560504643982-286628ae-43e1-4995-9d56-846ed6185d13.svg#align=left&display=inline&height=13&originHeight=13&originWidth=11&size=0&status=done&width=11) 编码再解码成真实图片，判别器 D 用到的是作者自己提出来的条件判别器 PatchGAN ，判别器 D 的作用是在轮廓图 ![](https://cdn.nlark.com/yuque/0/2019/svg/307794/1560504643983-469a59d8-cd1a-475f-9fd7-2591d14df223.svg#align=left&display=inline&height=13&originHeight=13&originWidth=11&size=0&status=done&width=11) 的条件下，对于生成的图片 ![](https://cdn.nlark.com/yuque/0/2019/svg/307794/1560565027960-49b9be2d-65e7-41e2-91cb-db8523301dc9.svg#align=left&display=inline&height=23&originHeight=23&originWidth=40&size=0&status=done&width=40) 判断为假，对于真实图片判断为真。

<a name="2cffab5b"></a>
#### 2.3 cGAN 与 Pix2pix 对比

![](https://cdn.nlark.com/yuque/0/2019/png/307794/1560565027847-99f4a0d6-4ba5-4051-b069-7452e504db19.png#align=left&display=inline&height=398&originHeight=643&originWidth=808&size=0&status=done&width=500)

<a name="95836c9a"></a>
#### 2.4 损失函数

一般的 cGANs 的目标函数如下：

$L_{cGAN}(G, D) =E_{x,y}[log D(x, y)]+E_{x,z}[log(1 − D(x, G(x, z))]$

其中 G 试图最小化目标而 D 则试图最大化目标，即：$\rm G^∗ =arg; min_G; max_D ;L_{cGAN}(G, D)$

为了做对比，同时再去训练一个普通的 GAN ，即只让 D 判断是否为真实图像。

$\rm L_{cGAN}(G, D) = E_y[log D(y)]+ E_{x,z}[log(1 − D(G(x, z))]$

对于图像翻译任务而言，G 的输入和输出之间其实共享了很多信息，比如图像上色任务、输入和输出之间就共享了边信息。因而为了保证输入图像和输出图像之间的相似度、还加入了 L1 Loss：

$\rm L_{L1}(G) = E_{x,y,z}[||y − G(x, z)||_1] $

即**生成的 fake images 与 真实的 real images 之间的 L1 距离，**（imgB**'** 和imgB）保证了输入和输出图像的相似度。

最终的损失函数：

$\rm G^∗ = arg;\underset{G}{min};\underset{D}{max}; L_{cGAN}(G, D) + λL_{L1}(G)$

<a name="efd82e98"></a>
### 3.网络架构（网络体系结构）

生成器和判别器都使用模块 convolution-BatchNorm-ReLu

<a name="f6b8ac58"></a>
#### 3.1 生成网络G

图像到图像翻译问题的一个定义特征是它们将高分辨率输入网格映射到高分辨率输出网格。 另外，对于我们考虑的问题，输入和输出的表面外观不同，但两者应该共享一些信息。 因此，输入中的结构与输出中的结构大致对齐。 我们围绕这些考虑设计了生成器架构。

![](https://cdn.nlark.com/yuque/0/2019/png/307794/1560565028107-13bae6fd-1a4a-4d79-b38a-a7691b1beb14.png#align=left&display=inline&height=165&originHeight=248&originWidth=753&size=0&status=done&width=500)

U-Net 结构基于 Encoder-Decoder 模型，而 Encoder 和 Decoder 是对称结构。 U-Net 的不同之处是将第 i 层和第 n-i 层连接起来，其中 n 是层的总数，这种连接方式称为跳过连接（skip connections）。第 i 层和第 n-i 层的图像大小是一致的，可以认为他们承载着类似的信息 。

<a name="6a4b2dc4"></a>
#### 3.2 判别网络 D

用损失函数 L1 和 L2 重建的图像很模糊，也就是说L1和L2并不能很好的恢复图像的高频部分(图像中的边缘等)，但能较好地恢复图像的低频部分(图像中的色块)。

> 图像的高低频是对图像各个位置之间强度变化的一种度量方法，低频分量：主要对整副图像的强度的综合度量。高频分量：主要是对图像边缘和轮廓的度量。如果一副图像的各个位置的强度大小相等,则图像只存在低频分量,从图像的频谱图上看,只有一个主峰,且位于频率为零的位置。如果一副图像的各个位置的强度变化剧烈，则图像不仅存在低频分量，同时也存在多种高频分量，从图像的频谱上看，不仅有一个主峰，同时也存在多个旁峰。


为了能更好得对图像的局部做判断，Pix2pix 判别网络采用 patchGAN 结构，也就是说把图像等分成多个固定大小的 Patch，分别判断每个Patch的真假，最后再取平均值作为 D 最后的输出。这样做的好处：

- D  的输入变小，计算量小，训练速度快。
- 因为 G 本身是全卷积的，对图像尺度没有限制。而D如果是按照Patch去处理图像，也对图像大小没有限制。就会让整个 Pix2pix 框架对图像大小没有限制，增大了框架的扩展性。

论文中将 PatchGAN 看成另一种形式的纹理损失或样式损失。在具体实验时，采用不同尺寸的 patch，发现 70x70 的尺寸比较合适。

<a name="770cadf8"></a>
#### 3.3 优化和推理

训练使用的是标准的方法：交替训练 D 和 G；并使用了 minibatch SGD 和 Adam 优化器。

在推理的时候，我们用训练阶段相同的方式来运行生成器。在测试阶段使用 dropout 和 batch normalization，这里我们使用 test batch 的统计值而不是 train batch 的。

<a name="ac096468"></a>
### 4.源码解读

该部分主要是解读论文源码：[https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 。

![](https://cdn.nlark.com/yuque/0/2019/png/307794/1560504644207-8c658dc6-b5b5-4b90-b490-4f771a80d099.png#align=left&display=inline&height=1338&originHeight=495&originWidth=185&size=0&status=done&width=500)

- 文件 train：

通用的训练脚本，可以通过传参指定训练不同的模型和不同的数据集。

`--model`: e.g.,`pix2pix`,`cyclegan`,`colorization`

`--dataset_mode`: e.g.,`aligned`,`unaligned`,`single`,`colorization`)

- 文件test:

通用的测试脚本，通过传参来加载模型 `-- checkpoints_dir`，保存输出的结果 `--results_dir`。

<a name="74e4ca60"></a>
#### 4.1 文件夹data：

该目录中的文件包含数据的加载和处理以及用户可制作自己的数据集。下面详细说明data下的文件：

- **`__init__.py`:** 实现包和train、test脚本之间的接口。train.py 和 test.py 根据给定的 opt 选项调包来创建数据集 `from data import create_dataset`和`dataset = create_dataset(opt)`
- **`base_dataset.py`:**继承了 torch 的 dataset 类和抽象基类，该文件还包括了一些常用的图片转换方法，方便后续子类使用。
- **`image_folder.py`:**更改了官方pytorch的image folder的代码，使得从当前目录和子目录都能加载图片。
- **`template_dataset.py`:**为制作自己数据集提供了模板和参考，里面注释一些细节信息。
- **`aligned_dataset.py`** 和 **`unaligned_dataset.py`:**区别在于前者从同一个文件夹中加载的是一对图片 {A,B} ，后者是从两个不同的文件夹下分别加载 {A},{B} 。
- **`single_dataset.py`:**只加载指定路径下的一张图片。
- **`colorization_dataset.py`:**加载一张 RGB 图片并转化成（L，ab）对在 Lab 彩色空间，pix2pix用来绘制彩色模型。

<a name="ea8640c7"></a>
#### 4.2 文件夹**models：**

models 包含的模块有：目标函数，优化器，网络架构。下面详细说明models下的文件：

- **`__init__.py`:** 为了实现包和train、test脚本之间的接口。`train.py` 和 `test.py` 根据给定的 opt 选项调包来创建模型 `from models import create_model` 和 `model = create_model(opt)`。
- **base_model.py:** 继承了抽象类，也包括一些其他常用的函数：`setup`、`test`、`update_learning_rate`、`save_networks`、`load_networks`，在子类中会被使用。
- **template_model.py:** 实现自己模型的一个模板，里面注释了一些细节。
- **pix2pix_model.py:** 实现了pix2pix 模型，模型训练数据集`--dataset_mode aligned`，默认情况下`--netG unet256 --netD basic`discriminator (PatchGAN)。 `--gan_mode vanilla`GAN loss (标准交叉熵)。
- **colorization_model.py:**继承了pix2pix_model,模型所做的是：将黑白图片映射为彩色图片。`-dataset_model colorization` dataset。默认情况下，`colorization` dataset会自动设置`--input_nc 1`and`--output_nc 2`。
- **cycle_gan_model.py:**来实现cyclegan模型。`--dataset_mode unaligned`dataset，`--netG resnet_9blocks`ResNet generator，`--netD basic`discriminator (PatchGAN introduced by pix2pix)，a least-square GANs[objective](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1611.04076)(`--gan_mode lsgan` )
- **networks.py:**包含生成器和判别器的网络架构，normalization layers,初始化方法，优化器结构（learning rate policy）GAN的目标函数（`vanilla`,`lsgan`,`wgangp`）。
- **test_model.py:**用来生成cyclegan的结果，该模型自动设置`--dataset_mode single`。

<a name="2e1f3497"></a>
#### 4.3 文件夹**options:**

包含训练模块，测试模块的设置`TrainOptions和TestOptions`都是 `BaseOptions`的子类。详细说明options下的文件。

- **__init__.py:**该文件起到让python解释器将options文件夹当做包来处理。
- **base_options.py:**除了training,test都用到的option,还有一些helper 方法：parsing,printing,saving options。
- **train_options.py:**训练需要的options。
- **test_options.py**:测试需要的options。

<a name="9c23ce96"></a>
#### 4.4 文件夹**utils：**

主要包含一些有用的工具，如数据的可视化。详细说明utils下的文件：

- **__init__.py:**该文件起到让python解释器将utils文件夹当做包来处理。
- **get_data.py:**用来下载数据集的脚本。
- **html.py:**保存图片写成html。基于diminate中的DOM API。
- **image_pool.py:**实现一个缓冲来存放之前生成的图片。
- **visualizer.py:**保存图片，展示图片。
- **utils.py:**包含一些辅助函数：tensor2numpy转换，mkdir诊断网络梯度等。

<a name="9b8892b3"></a>
### 5. 总结与展望

<a name="e212c908"></a>
#### 5.1 pix2pix的优缺点

Pix2pix模型是 ![](https://cdn.nlark.com/yuque/0/2019/svg/307794/1560504644106-23df7369-ef17-4711-acf9-593cc9746b6d.svg#align=left&display=inline&height=13&originHeight=13&originWidth=11&size=0&status=done&width=11) **到** ![](https://cdn.nlark.com/yuque/0/2019/svg/307794/1560504644145-758ec7f9-23f5-4452-b877-980734c43502.svg#align=left&display=inline&height=16&originHeight=16&originWidth=9&size=0&status=done&width=9) **之间的一对一映射**。也就说，pix2pix就是对ground truth的重建：输入轮廓图→经过Unet编码解码成对应的向量→解码成真实图。这种一对一映射的应用范围十分有限，当我们输入的数据与训练集中的数据差距较大时，生成的结果很可能就没有意义，这就要求我们的数据集中要尽量涵盖各种类型。

本文将Pix2Pix论文中的所有要点都表述了出来，主要包括：

- cGAN，输入为图像而不是随机向量
- U-Net，使用skip-connection来共享更多的信息
- Pair输入到D来保证映射
- Patch-D来降低计算量提升效果
- L1损失函数的加入来保证输入和输出之间的一致性

<a name="cece3051"></a>
#### 5.2 总结

目前，您可以在  [Mo ](**momodel.cn**) 平台的应用中心中找到 [pix2pixGAN](http://www.momodel.cn:8899/appcenter/5c0cb5df1afd945819064752)，可以体验论文实验部分图像建筑标签→照片（ Architectural labels→photo），即将**您绘制的建筑图片草图生成为你心目中的小屋** 。您在学习的过程中，遇到困难或者发现我们的错误，可以随时联系我们。

通过本文，您应该初步了解Pix2pix模型的网络结构和实现原理，以及关键部分代码的初步实现。如果您对深度学习tensorflow比较了解，可以参考[tensorflow版实现Pix2pix](https://github.com/yenchenlin/pix2pix-tensorflow)；如果您对pytorch框架比较熟悉，可以参考[pytorch实现Pix2pix](https://github.com/phillipi/pix2pix)；如果您想更深入的学习了解starGAN原理，可以参考[论文](https://arxiv.org/pdf/1611.07004.pdf)。

<a name="d96e87ae"></a>
### 6.参考：

1.论文：[https://arxiv.org/pdf/1611.07004.pdf](https://arxiv.org/pdf/1611.07004.pdf)

2.Pix2pix官网:[https://phillipi.github.io/pix2pix/](https://phillipi.github.io/pix2pix/)

3.代码PyTorch版本：[https://github.com/phillipi/pix2pix](https://github.com/phillipi/pix2pix)

4.代码tensorflow版本：[https://github.com/yenchenlin/pix2pix-tensorflow](https://github.com/yenchenlin/pix2pix-tensorflow)

5.代码tensorflow版本：[https://github.com/affinelayer/pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow)

6.知乎：[https://zhuanlan.zhihu.com/p/38411618](https://zhuanlan.zhihu.com/p/38411618)

7.知乎：[https://zhuanlan.zhihu.com/p/55059359](https://zhuanlan.zhihu.com/p/55059359)

8.博客：[https://blog.csdn.net/qq_16137569/article/details/79950092](https://blog.csdn.net/qq_16137569/article/details/79950092)

9.博客：[https://blog.csdn.net/infinita_LV/article/details/85679195](https://blog.csdn.net/infinita_LV/article/details/85679195)

10.博客：[https://blog.csdn.net/weixin_36474809/article/details/89004841](https://blog.csdn.net/weixin_36474809/article/details/89004841)

<a name="oxtTX"></a>
### 关于我们
**Mo**（网址：[**momodel.cn**](http://www.momodel.cn:8899/)）是一个支持 Python 的**人工智能在线建模平台**，能帮助你快速开发、训练并部署模型。

---

**Mo 人工智能俱乐部** 是由网站的研发与产品设计团队发起、致力于降低人工智能开发与使用门槛的俱乐部。团队具备大数据处理分析、可视化与数据建模经验，已承担多领域智能项目，具备从底层到前端的全线设计开发能力。主要研究方向为大数据管理分析与人工智能技术，并以此来促进数据驱动的科学研究。

目前俱乐部每周六在杭州举办以机器学习为主题的线下技术沙龙活动，不定期进行论文分享与学术交流。希望能汇聚来自各行各业对人工智能感兴趣的朋友，不断交流共同成长，推动人工智能民主化、应用普及化。<br />![image.png](https://cdn.nlark.com/yuque/0/2019/png/307794/1560565564936-bcd9ec1e-8e47-4373-ba0d-1ab3a696aee4.png#align=left&display=inline&height=175&name=image.png&originHeight=349&originWidth=720&size=170790&status=done&width=360)

