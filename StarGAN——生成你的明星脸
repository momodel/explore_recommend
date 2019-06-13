# StarGAN——生成你的明星脸

<a name="8a359fe2"></a>
### 1 GAN 介绍

	GAN，叫做生成对抗网络 (**Generative Adversarial Network**) 。其基本原理是生成器网络 G(**Generator**) 和判别器网络 D(**Discriminator**) 相互博弈。生成器网络 G 的主要作用是生成图片，在输入**一个随机编码 (random code) z**后，自动的生成假样本 G(z) 。判别器网络 D 的主要作用是判断输入是否为真实样本并提供反馈机制，真样本则输出 1 ，反之为 0 。在两个网络相互博弈的过程中，两个网络的能力都越来越高：G 生成的图片越来越像真样本，D 也越来越会判断图片的真假，然后我们在最大化 D 的前提下，最小化 D 对 G 的判断能力，这实际上就是最小最大值问题，或者说二人零和博弈，其目标函数表达式：<br />![](https://cdn.nlark.com/yuque/__latex/449b9d0db058f4414c75510f1f4ac9fa.svg#card=math&code=%5Crm%20%5Cunderset%7BG%7D%7Bmin%7D%20%5C%3B%20%5Cunderset%7BD%7D%7Bmax%7D%20%5C%3B%20E%5Blog%20D%28G%28z%29%29%2Blog%281-D%28x%29%29%5D&height=32&width=280)<br />其中表达式中的第一项 D(G(z)) 处理的是假图像 G(z) ，我们尽量降低评分 D(G(z)) ；第二项处理的是真图像 x ，此时评分要高。但是 GAN 并不是完美的，也有自己的局限性。比如说没有用户控制的能力和低分辨率与低质量的问题。

	为了提高 GAN 的用户控制能力，人类进行了一些列的探索研究。比如 Pix2Pix 模型采用有条件的使用用户输入，使用**成对的数据 (paired data) ** 进行训练； CycleGAN 模型使用**不成对的数据 (unpaired data) **的就能训练 。但无论是 Pix2Pix 还是 CycleGAN ，都是解决了从一个领域到另一个领域的图像转换问题。当有很多领域需要转换时，对于每一个领域转换，都需要重新训练一个模型去解决。目前，存在的模型处理多领域图像生成任务时，学习 k 个领域之间所有映射就必须训练 k * (k-1) 个生成器。如果训练一对一的图像多领域生成任务时，主要会导致两个问题：

- 训练低效，每次训练耗时很大。
- 训练效果有限，因为一个领域转换单独训练的话就不能利用其它领域的数据来增大泛化能力。

![](https://cdn.nlark.com/yuque/0/2019/png/307794/1559194787734-3ac9b5ef-d8ee-404d-ab0e-a51f41f57a82.png#align=left&display=inline&height=311&originHeight=311&originWidth=560&size=0&status=done&width=560)

上图中 (a) 模型说明如何训练 12 个不同生成器网络以达到 4 个不同领域图像之间转换任务。很明显每个生成器不能够充分利用整个训练数据，只能从 4 个领域中 2 个领域相互学习，这样就会生成图片质量不好。而上图（b）中的模型就可以解决这些问题，该模型接受多个领域训练数据，并仅使用一个生成器来学习多领域图像之间映射关系。根据模型的长相将该模型称为星形网络，外文名就是 StarGAN 。

![](https://cdn.nlark.com/yuque/0/2019/png/307794/1559194787748-4c1d68ab-fbc1-49c5-8e74-53ab304607a2.png#align=left&display=inline&height=520&originHeight=520&originWidth=1142&size=0&status=done&width=1142)<br />上图是根据 StarGAN 模型训练出的效果。在同一种模型下，可以做多领域图像之间的转换，比如更换头发颜色、更换表情、更换年龄等。

<a name="0572c407"></a>
### 2 StarGAN模型及其优点

<a name="f5559b43"></a>
#### 2.1 starGAN介绍

![](https://cdn.nlark.com/yuque/0/2019/png/307794/1559194787795-f87eab09-9d77-4f46-996e-bc142c05f0cf.png#align=left&display=inline&height=464&originHeight=464&originWidth=1149&size=0&status=done&width=1149)

	上图是对 StarGAN 的简单介绍，主要包含判别器 D 和生成器 G 。<br />（a）D 对真假图片进行判别，真图片判真，假图片判假，真图片被分类到相应域。<br />（b）G 接受真图片和目标域标签并生成假图片；<br />（c）G 在给定原始域标签的情况下将假图片重建为原始图片（重构损失）;<br />（d）G 尽可能生成与真实图像无法区分的图像，并且通过 D 分类到目标域。

<a name="fc25af8c"></a>
#### 2.2 StarGAN 优点

- 提出 StarGAN 网络模型，仅使用一个 G 和 D 就可以实现多个领域之间图像生成和训练。
- 采用 mask vector 方法控制所有可用域图像标签以实现训练集之间的多领域图像转换。
- StarGAN 相对于基准模型, 在面部属性转移和面部表情合成的任务中有更好的效果 (具体数据请参看原论文中的实验部分)

<a name="572d8709"></a>
### 3 StarGAN

	首先描述 StarGAN 网络，在一个数据集中进行多领域的图像转换任务；然后我们讨论了如何使 StarGAN 能合并包含不同标签的数据集以及对其中任意的标签属性灵活进行图像转换。

<a name="ab9c604c"></a>
#### 3.1 多领域图像转换

	训练一个生成器 G ，能够多领域映射。将带有领域标签 c 的输入图像 x 转换为输出图像 y，即![](https://cdn.nlark.com/yuque/__latex/e68c90ce84aff41543e197e18e2c9c44.svg#card=math&code=%5Crm%20G%28x%EF%BC%8Cc%29%20%20%5Crightarrow%20%20y&height=26&width=90) 。随机生成目标领域标签 c 使得 G 能够灵活的转换输入图像，同时使用 D 控制多领域。这样 D 就在图像源和域标签上产生概率分布，即![](https://cdn.nlark.com/yuque/__latex/24c1e0750a0a24356e998d31628d9f93.svg#card=math&code=%5Crm%20D%20%3A%20x%20%E2%86%92%20%7BD%20src%20%28x%29%2CD%20cls%20%28x%29%7D&height=24&width=173)。

<a name="0c8db0b5"></a>
##### 3.1.1 对抗损失函数 (Adversarial Loss)

	使用对抗损失函数提高生成图像质量，达到 D 无法区分出来输出图像和生成图像之间的差别：<br />![](https://cdn.nlark.com/yuque/__latex/fd464f95747e3d3a08c5ae9d00663645.svg#card=math&code=L_%7Badv%7D%3DE_x%5BlogD_%7Bsrc%7D%28x%29%5D%20%2B%20E_%7Bx%2Cc%7D%5Blog%281%20-%20D_%7Bsrc%7D%28G%28x%EF%BC%8Cc%29%29%5D&height=26&width=367)<br />	根据输入图像 x 和目标领域标签 c ，由 G 生成输出图像![](https://cdn.nlark.com/yuque/__latex/245564aa1876ac2ce3b5255b63dad946.svg#card=math&code=%5Crm%20G%28x%EF%BC%8Cc%29&height=26&width=57)，同时 D 区分出真实图像和生成图像。将![](https://cdn.nlark.com/yuque/__latex/b1f6cc3ed65efd22b77d43faefa7cab9.svg#card=math&code=%5Crm%20D_%7Bsrc%7D%28x%29&height=24&width=49)<br />作为输入图像 x 经过 D 之后得到的可能性分布。生成器 G 使这个式子尽可能的小，而 D 则尽可能使其最大化。

<a name="8f7972d8"></a>
##### 3.1.2 目标域分类损失函数(Domain Classification Loss)

	对于一个输入图像 x 和目标分布标签 c ，我们的目标是将 x 转换为输出图像 y后能够被正确分类为目标分布 c 。为了实现这一目标，我们在 D 之上添加一个辅助分类器，并在优化 G 和 D 时采用目标域分类损失函数。简单来说，我们将这个式子分解为两部分：一个真实图像的分布分类损失用于约束 D ，一个假图像的分布分类损失用于约束 G 。其表达式如下所示：<br />![](https://cdn.nlark.com/yuque/__latex/c7172e2fa6a79af8eef1a4fc56bf3d36.svg#card=math&code=L%5E%7Br%7D_%7Bcls%7D%20%3D%20E_%7Bx%2Cc%E2%80%99%7D%20%5B-logD_%7Bcls%7D%28c%E2%80%99%7Cx%29%5D&height=25&width=190)<br />其中，![](https://cdn.nlark.com/yuque/__latex/fc6c081e14d3e6087a4ab16141c79af0.svg#card=math&code=D_%7Bcls%7D%28c%E2%80%99%7Cx%29&height=24&width=67)代表 D 计算出来的领域标签的可能性分布。一方面，通过将这个式子最小化， D 将真实图像 x 正确分类到与其相关分布 c' 。另一方面，假图像的分类分布的损失函数定义如下：<br />![](https://cdn.nlark.com/yuque/__latex/be80ae7e0115de058574d0bd31924ff7.svg#card=math&code=L_%7Bcls%7D%5Ef%20%3D%20E_%7Bx%2Cc%7D%5B-log%20D_%7Bcls%7D%28c%7CG%28x%EF%BC%8Cc%29%29%5D&height=26&width=229)<br />即 G 使这个式子最小化，使得生成的图像能够被 D 判别为目标领域 c。

<a name="25165474"></a>
##### 3.1.3 重构误差(Reconstruction Loss)

	通过最小化对抗损失和分类损失， G 训练生成的图像尽可能与真实图像一样，并且能够被分类到正确的目标领域。然而，最小化这两个损失函数不能保证 , 转换后的图像中，只改变领域差异的部分, 而保留输入图像中的其他内容 。故对 G 使用循环一致性损失函数 (cycle consistency loss) ，如下：<br />![](https://cdn.nlark.com/yuque/__latex/e49bd9efdb693a94a1e4e104cde09060.svg#card=math&code=L_%7Brec%7D%20%3D%20E_%7Bx%2Cc%2Cc%E2%80%99%7D%20%5B%7C%7Cx%20-%20G%28G%28x%2Cc%29%2Cc%E2%80%99%29%7C%7C_%7B1%7D%5D&height=25&width=247)<br />其中： G 以生成图像 G(x，c) 以及原始输入图像领域标签 c' 为输入，努力重构出原始图像 x 。我们选择L范数作为重构损失函数。注意到我们两次使用了同一个生成器，第一次将原始图像转换到目标领域的图像，然后将生成的图像重构回原始图像。

<a name="c52ff4c7"></a>
##### 3.1.4 总体损失函数表示(Full Objective)

最终 G 和 D 的损失函数表示如下：<br />![](https://cdn.nlark.com/yuque/__latex/28539c1a3a87690ff222aba9ee7b3009.svg#card=math&code=L_D%20%3D%20-L_%7Badv%7D%20%2B%20%5Clambda_%7Bcls%7DL%5E%7Br%7D_%7Bcls%7D&height=25&width=158)<br />![](https://cdn.nlark.com/yuque/__latex/25fcabc3be289d2e21060ba3406eade5.svg#card=math&code=L_G%20%3D%20L_%7Badv%7D%20%2B%20%5Clambda_%7Bcls%7DL%5E%7Bf%7D_%7Bcls%7D%2B%20%5Clambda_%7Brec%7DL_%7Brec%7D&height=25&width=220)<br />其中 ![](https://cdn.nlark.com/yuque/__latex/58a7b8d99ac1deda150789c823d72170.svg#card=math&code=%5Clambda_%7Bcls%7D&height=24&width=25)_ 和 ![](https://cdn.nlark.com/yuque/__latex/e226236c75e6168e6a38f132609e96e1.svg#card=math&code=%5Clambda_%7Brec%7D&height=24&width=27)_ 是控制分类误差和重构误差相对于对抗误差的相对权重的超参数。在所有实验中，我们设置![](https://cdn.nlark.com/yuque/__latex/1b8b59c0473bbcbc19019cc228483260.svg#card=math&code=%5Clambda_%7Bcls%7D%20%3D%201%2C%5Clambda_%7Brec%7D%20%3D%2010&height=24&width=127)。

<a name="27f07a07"></a>
##### 3.1.5 改进损失函数

	为了 GAN 训练过程稳定，生成高质量的图像，论文中采用自定义梯度惩罚来代替对抗误差损失：<br />![](https://cdn.nlark.com/yuque/__latex/429278eee56457694bc018efcab9697a.svg#card=math&code=L_%7Badv%7D%3DE_x%5BD_%7Bsrc%7D%28x%29%5D%20-%20E_%7Bx%2Cc%7D%5BD_%7Bsrc%7D%28G%28x%EF%BC%8Cc%29%29%5D%20-%20%5Clambda_%7Bgp%7DE_%7B%5Chat%7Bx%7D%7D%20%5B%28%7C%7C%5Cnabla%7B%5Chat%7Bx%7D%7DD_%7Bsrc%7D%28%5Chat%7Bx%7D%29%7C%7C_%7B2%7D-1%29%5E2%5D&height=26&width=512)<br />其中：![](https://cdn.nlark.com/yuque/__latex/2a95aaaf954c2187999c6357b04a58dd.svg#card=math&code=%5Chat%7Bx%7D&height=24&width=9) 表示真实和生成图像之间均匀采样的直线，试验时![](https://cdn.nlark.com/yuque/__latex/b910dfe437e725b3073ef930e4fc5b98.svg#card=math&code=%5Clambda_%7Bgp%7D%3D10&height=25&width=61)。

<a name="f1c068f3"></a>
#### 3.2 多数据集训练

	starGAN 的一个重要优势在于它能够同时合并包含不同标签的不同数据集，使得其在测试阶段能够控制所有的标签。从多个数据集学习的问题在于标签信息对每一个数据集而言只是部分已知。在 CelebA 和 RaFD 的例子中，前一个数据集包含诸如发色，性别等信息，但它不包含任何后一个数据集中包含的诸如开心生气等表情标签。这会引起问题，因为在将 G(x，c) 重构回输入图像 x 时需要完整的标签信息 c' 。

<a name="1f4900a3"></a>
##### 3.2.1 向量掩码(Mask Vector)

	为了缓解这一问题，我们引入了向量掩码 m，使 StarGAN 模型能够忽略不确定的标签，专注于特定数据集提供的明确的已知标签。在 StarGAN 中我们使用 n 维的 one-hot 向量来代表 m ，n 表示数据集的数量。除此之外，我们将标签的同一版本定义为一个数组：<br />![](https://cdn.nlark.com/yuque/__latex/61920403581a295b4eb0fe6511f7821a.svg#card=math&code=%5Crm%20%5Coverline%7Bc%7D%20%20%3D%20%5Bc_1%EF%BC%8C%E2%80%A6%EF%BC%8Cc_n%EF%BC%8Cm%5D&height=26&width=152)<br />其中：[·]表示串联，其中 c表示第 i 个数据集的标签，已知标签 c 的向量能用二值标签表示二值属性或者用 one-hot 的形式表示多类属性。对于剩下的 n-1 个未 i 知标签我们简单的置为 0 。

<a name="af563e4c"></a>
##### 3.2.2 训练策略

	利用多数据集训练 StarGAN 时，我们使用上面定义的![](https://cdn.nlark.com/yuque/__latex/cfd07535363876101cb481b7daa29a1d.svg#card=math&code=%5Coverline%7Bc%7D&height=24&width=8) 作为生成器的输入。如此，生成器学会忽略非特定的标签，而专注于指定的标签。除了输入标签![](https://cdn.nlark.com/yuque/__latex/cfd07535363876101cb481b7daa29a1d.svg#card=math&code=%5Coverline%7Bc%7D&height=24&width=8) ，此处的生成器与单数据集训练的生成器网络结构一样。另一方面我们也扩展判别器的辅助分类器的分类类别到到所属聚集的所有标签。最后，我们将我们的模型按照多任务学习的方式进行训练，其中，判别器只将已知标签相关的分类误差最小化即可。

<a name="1f34d00e"></a>
#### 3.3 训练数据处理

以 celebA 数据为例，下载后的数据包括 label 文件和图像。

- 文件的第一行为图像的总数：202599。
- 第二行为数据处理的类别，共 40 种，如下：
> (1， '5_o_Clock_Shadow')， (2， 'Arched_Eyebrows')， (3， 'Attractive')， (4， 'Bags_Under_Eyes')， (5， 'Bald')， (6， 'Bangs')， (7， 'Big_Lips')， (8， 'Big_Nose')， (9， 'Black_Hair')， (10， 'Blond_Hair')， (11， 'Blurry')， (12， 'Brown_Hair')， (13， 'Bushy_Eyebrows')， (14， 'Chubby')， (15， 'Double_Chin')， (16， 'Eyeglasses')， (17， 'Goatee')， (18， 'Gray_Hair')， (19， 'Heavy_Makeup')， (20， 'High_Cheekbones')， (21， 'Male')， (22， 'Mouth_Slightly_Open')， (23， 'Mustache')， (24， 'Narrow_Eyes')， (25， 'No_Beard')， (26， 'Oval_Face')， (27， 'Pale_Skin')， (28， 'Pointy_Nose')， (29， 'Receding_Hairline')， (30， 'Rosy_Cheeks')， (31， 'Sideburns')， (32， 'Smiling')， (33， 'Straight_Hair')， (34， 'Wavy_Hair')， (35， 'Wearing_Earrings')， (36， 'Wearing_Hat')， (37， 'Wearing_Lipstick')， (38， 'Wearing_Necklace')， (39， 'Wearing_Necktie')， (40， 'Young')


- 第三行及之后的每行为，图像名，已经对应的 40 种类别的 label ， label 值为 1 或 -1。
> 000001.jpg -1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 -1 -1 1 1 -1 1 -1 -1 1 -1 -1 1 -1 -1 -1 1 1 -1 1 -1 1 -1 -1 1


<a name="d95db800"></a>
### 4 总结与展望

通过本文学习，您应该初步了解 StarGAN 模型的网络结构和实现原理，以及关键部分代码的初步实现。如果您对深度学习 Tensorflow 比较了解，可以参考 [Tensorflow版实现starGAN](https：//github.com/taki0112/StarGAN-Tensorflow/)；如果您对pytorch框架比较熟悉，可以参考 [pytorch实现starGAN](https：//github.com/yunjey/StarGAN/)；如果您想更深入的学习了解starGAN原理，可以参考 [论文](https：//arxiv.org/pdf/1711.09020.pdf)。

如果想体验项目效果，您可以登陆  [Mo 平台](http://www.momodel.cn:8899/)，在 [应用中心](http：//www.momodel.cn：8899/appcenter) 中找到 [StarGAN](http://www.momodel.cn:8899/appcenter/5c0cc4591afd945c5177fb51)，可以体验以下五种特征['Black_Hair'， 'Blond_Hair'， 'Brown_Hair'， 'Male'， 'Young'] 的风格变换。考虑到代码较长，我们在[StarGAN 项目源码](http://www.momodel.cn:8899/explore/5c0cc4591afd945c5177fb51?type=app)中对相关代码做了详细解释。您在学习的过程中，遇到困难或者发现我们的错误，可以随时联系我们。

<a name="VGhEq"></a>
### 5 参考资料
1.论文：https：[//arxiv.org/pdf/1711.09020.pdf](https://arxiv.org/pdf/1711.09020.pdf) <br />2.博客：https：[//blog.csdn.net/stdcoutzyx/article/details/78829232](https://blog.csdn.net/stdcoutzyx/article/details/78829232)<br />3.博客：https：[//www.cnblogs.com/Thinker-pcw/p/9785379.html](https://www.cnblogs.com/Thinker-pcw/p/9785379.html)<br />4.pytorch原版github地址：https：[//github.com/yunjey/StarGAN](https://github.com/yunjey/StarGAN)<br />5.tensorflow版github地址：https：[//github.com/taki0112/StarGAN-Tensorflow](https://github.com/taki0112/StarGAN-Tensorflow)<br />6.Celeba数据集：[https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0](https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0)

---

**Mo**（网址：**momodel.cn**）是一个支持 Python 的**人工智能在线建模平台**，能帮助你快速开发、训练并部署模型。

---

**Mo 人工智能俱乐部** 是由网站的研发与产品设计团队发起、致力于降低人工智能开发与使用门槛的俱乐部。团队具备大数据处理分析、可视化与数据建模经验，已承担多领域智能项目，具备从底层到前端的全线设计开发能力。主要研究方向为大数据管理分析与人工智能技术，并以此来促进数据驱动的科学研究。

目前俱乐部每周六在杭州举办以机器学习为主题的线下技术沙龙活动，不定期进行论文分享与学术交流。希望能汇聚来自各行各业对人工智能感兴趣的朋友，不断交流共同成长，推动人工智能民主化、应用普及化。

![双二维码.png](https://cdn.nlark.com/yuque/0/2019/png/307794/1559194880763-378ee5d1-5cab-4b92-b257-33f4c6d624c1.png?x-oss-process=image/format,png#align=left&display=inline&height=540&name=%E5%8F%8C%E4%BA%8C%E7%BB%B4%E7%A0%81.png&originHeight=540&originWidth=1114&size=2407242&status=done&width=1114)
<a name="d17a0f0b"></a>
### 
<a name="2Zgwc"></a>
### 
