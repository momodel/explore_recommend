# Mo-AI 俱乐部第 41 期机器学习学术沙龙邀请函

Mo 人工智能俱乐部 正式向感兴趣的小伙伴们发出诚挚的邀请！5月17日（周日），我们将在线上举办第 41 期机器学习学术沙龙。

+ 活动主题：Universal Representation Learning  
+ 活动举办方：Mo人工智能俱乐部
+ 活动方式：线上（Mo平台直播&钉钉直播)
+ 活动时间：5月17日19：30-21：30


### 活动内容
#### 主讲人
李则熹(浙江大学-在校学生-生物系统工程与食品科学学院-本科生-农业工程-农业工程1602)
#### 分享论文
Towards Universal Representation Learning for Deep Face Recognition

#### 大纲
1. 背景介绍
2. 通用表示学习算法介绍
3. 实验结果
4. 讨论与展望

#### 引言
该文章发表在CVPR2020，提出了一种针对人脸识别的通用表示学习算法框架（universal representation learning)，并且该框架也可以被应用到其他的视觉识别领域。

通常情况下，识别自然环境下的人脸是非常困难的，因为它们会出现各种各样的变化，比如脸的角度、遮挡、光线、低分辨率等。传统的方法一般通过引入新的变化数据来适应训练数据，将训练数据和一些变化综合在一起，如低分辨率、遮挡和脸的角度。然而，直接输入数据增强之后的训练数据（augmented data）并不会让网络很好地收敛，因为新引入的样本大多是困难样本（hard examples）。而这篇文章提出了一个通用的表示学习框架，它可以处理给定训练数据中可能产生的变化，而不需要利用目标领域的知识。作者将嵌入的特征（feature embedding）分解成多个子嵌入，并将每个子嵌入与不同置信值关联起来，使训练过程更加平滑。通过将变化分类损失和变化对抗损失在不同的分区上正则化，来进一步处理相关子嵌入，已获得更好的表示（representation）。实验表明，该方法在LFW和MegaFace等一般人脸识别数据集上取得了较好的性能，在TinyFace和IJB-S等极端基准上取得了较好的性能。

### 活动地址
钉钉扫码，加入直播群！
<img src="https://imgbed.momodel.cn/Picture1.png"   width="300">



<a name="jdNNk"></a>
### 机器学习沙龙回顾
| 期数 | 时间 | 内容 | 链接 |
| --- | --- | --- | --- |
| 1 | 3.16 | 【机器学习】初识机器学习；单变量线性回归<br /> | 吴恩达机器学习：[https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video](https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video)[（电脑端打开观看视频1-1到2-4）](https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video)<br /> |
| 2 | 3.23 | 【论文分享】DARTS；HCN 网络 | 论文分享资料：https://github.com/momodel/AIClub |
| 3 | 3.30 | 【机器学习】梯度下降；线性代数回顾 | 吴恩达机器学习：<br />[https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video](https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video)[（]()[]()[电脑端打开观看视频2-5到3-4](http://www.momodel.cn:8899/classroom/class?id=5c5696191afd94720cc94533&type=video)） |
| 4 | 4.13 | 【论文分享】Google Vizier；Metalearning；Block Federated Learning<br />【机器学习】线性代数回顾 + 多变量线性回归 | 论文分享资料：https://github.com/momodel/AIClub<br />吴恩达机器学习：<br />[https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video](https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video)（[电脑端打开观看视频3-5到4-4](http://www.momodel.cn:8899/classroom/class?id=5c5696191afd94720cc94533&type=video)） |
| 5 | 4.20 | 【机器学习】多变量线性回归+逻辑回归 | 吴恩达机器学习：<br />[https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video](https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video)（[电脑端打开观看视频4-5到5-5](http://www.momodel.cn:8899/classroom/class?id=5c5696191afd94720cc94533&type=video)）<br />课程直播链接：<br />[https://v.douyu.com/show/6Aw87OpDLojMYGkg](https://v.douyu.com/show/6Aw87OpDLojMYGkg) |
| 6 | 4.27 | 【论文分享】FastRNN+分布式去中心化优化算法+mobilenet<br />【机器学习】逻辑回归 + 正则化 | 论文分享资料：https://github.com/momodel/AIClub <br />吴恩达机器学习：<br />[https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video](https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video)（[电脑端打开观看视频5-6到6-3](http://www.momodel.cn:8899/classroom/class?id=5c5696191afd94720cc94533&type=video)）<br />课程直播链接：<br />[https://v.douyu.com/show/ZB5Kv9LlrJKWa93x](https://v.douyu.com/show/ZB5Kv9LlrJKWa93x) |
| 7 | 5.11 | 【项目实战】房价预测[](http://www.momodel.cn:8899/classroom/class?id=5c680b311afd943a9f70901b&type=practice) | 实战项目链接：[https://momodel.cn/classroom/class?id=5c680b311afd943a9f70901b&type=practice](https://momodel.cn/classroom/class?id=5c680b311afd943a9f70901b&type=practice)（电脑端打开“模型评价与验证-波士顿房价预测”进行实战操作） |
| 8 | 5.18 | 【论文分享】Maximal information coefficent+MAML+EIVHE<br />【机器学习】神经网络学习 | 论文分享资料：https://github.com/momodel/AIClub <br />吴恩达机器学习：[https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video](https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video)（[电脑端打开观看视频7-1到7-7](http://www.momodel.cn:8899/classroom/class?id=5c5696191afd94720cc94533&type=video)）课程直播链接：<br />[https://v.douyu.com/show/ZB5Kv9LjRGbWa93x](https://v.douyu.com/show/ZB5Kv9LjRGbWa93x) |
| 9 | 5.25 | 【机器学习】神经网络参数的反向传播算法  | 吴恩达机器学习：<br />[https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video](https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video)（[电脑端打开观看视频8-1到8-8](http://www.momodel.cn:8899/classroom/class?id=5c5696191afd94720cc94533&type=video)） |
| 10 | 6.1 | 【论文分享】神经网络剪枝技术；Road Monitor；蒙特卡罗方法<br />【机器学习】应用机器学习的建议和机器学习系统的设计 | 论文分享资料：https://github.com/momodel/AIClub <br /> 吴恩达机器学习：<br />[https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video](https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video)（[电脑端打开观看视频9-1到9-7](http://www.momodel.cn:8899/classroom/class?id=5c5696191afd94720cc94533&type=video)）<br />课程直播链接：[https://v.douyu.com/show/2Bj8vG5V0jQMObnd](https://v.douyu.com/show/2Bj8vG5V0jQMObnd)<br /> |
| 11 | 6.15 | 【论文分享】ArcFace+博弈论+AlexNet<br />【机器学习】机器学习系统设计 | 论文分享资料：https://github.com/momodel/AIClub<br />吴恩达机器学习：<br />[https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video](https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video)（[电脑端打开观看视频10-1到10-5](http://www.momodel.cn:8899/classroom/class?id=5c5696191afd94720cc94533&type=video)）<br />课程直播链接：<br />[https://v.douyu.com/show/DrwnvzANa80WPNaX](https://v.douyu.com/show/DrwnvzANa80WPNaX) |
| 12 | 6.22 | 【机器学习】支持向量机 | 吴恩达机器学习：<br />[https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video](https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video)（[电脑端打开观看视频11-1到11-6](http://www.momodel.cn:8899/classroom/class?id=5c5696191afd94720cc94533&type=video)）<br />课程直播链接：<br />[https://v.douyu.com/show/Qyz171Qawg57BJj9](https://v.douyu.com/show/Qyz171Qawg57BJj9) |
| 13 | 6.29 | 【论文分享】TensorFlow Federated 库使用；点击率预测；U-Net<br />【机器学习】无监督学习 | 论文分享资料：https://github.com/momodel/AIClub<br />吴恩达机器学习：<br />[https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video](https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video)（[电脑端打开观看视频12-1到12-5](http://www.momodel.cn:8899/classroom/class?id=5c5696191afd94720cc94533&type=video)）<br />课程直播链接：<br />[https://v.douyu.com/show/4xq3WDXPBjbWLGNz](https://v.douyu.com/show/4xq3WDXPBjbWLGNz) |
| 14 | 7.6 | 【项目实战】非监督学习—创建客户群 [](http://www.momodel.cn:8899/classroom/class?id=5c680b311afd943a9f70901b&type=practice) | 实战项目链接：[https://momodel.cn/classroom/class?id=5c680b311afd943a9f70901b&type=practice](https://momodel.cn/classroom/class?id=5c680b311afd943a9f70901b&type=practice)（电脑端打开“非监督学习—创建客户群”进行实战操作）<br />课程直播链接：[https://v.douyu.com/show/n8GzMXwZ4zg76qyP](https://v.douyu.com/show/n8GzMXwZ4zg76qyP) |
| 15 | 7.13 | 【论文分享】深度强化学习DQN；基于模型蒸馏技术的相互学习；数据蒸馏<br />【机器学习】降维 | 深度强化学习：[https://momodel.cn/live/5d286bfa1afd942ac31fbe40](https://momodel.cn/live/5d286bfa1afd942ac31fbe40)<br />基于模型蒸馏技术的相互学习：<br />[https://momodel.cn/live/5d2889011afd942b62a09731](https://momodel.cn/live/5d2889011afd942b62a09731)<br />数据蒸馏：<br />[https://momodel.cn/live/5d288aa61afd942b59249e95](https://momodel.cn/live/5d288aa61afd942b59249e95)<br />降维：<br />[https://momodel.cn/live/5d2896f91afd942aa22623f0](https://momodel.cn/live/5d2896f91afd942aa22623f0)<br />（电脑端打开链接，点击“进入课程”，点击“回放”按钮观看回放） |
| 16 | 7.20 | 【机器学习】异常检测 | 吴恩达机器学习：<br />[https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video](https://momodel.cn/classroom/class?id=5c5696191afd94720cc94533&type=video)（电脑端打开观看视频14-1到14-8） |
| 17 | 7.27 | 【论文分享】LocoaNet 网络；中文文本生成；联邦学习<br />【机器学习】推荐系统 | LocoaNet 网络：<br />https://momodel.cn/live/5d3ab0ac1afd9443b584dd84 <br />中文文本生成：<br />[https://momodel.cn/live/5d3af46a1afd94425f2c7f23](https://momodel.cn/live/5d3af46a1afd94425f2c7f23)<br />联邦学习：<br />[https://momodel.cn/live/5d3aea181afd944437eaf44b](https://momodel.cn/live/5d3aea181afd944437eaf44b)<br />推荐系统：<br />[https://momodel.cn/live/5d3859ad1afd94479ffa37d8](https://momodel.cn/live/5d3859ad1afd94479ffa37d8)<br />（电脑端打开链接，点击“进入课程”，点击“回放”按钮观看回放） |
| 18 | 8.3 | 【机器学习】大规模机器学习；应用举例：照片OCR（光学字符识别） | [https://momodel.cn/live/5d3fd87e1afd9432011487b2](https://momodel.cn/live/5d3fd87e1afd9432011487b2)<br />（电脑端打开链接，点击“进入课程”，点击“回放”按钮观看回放） |
| 19 | 8.10 | 【论文分享】交通量预测;Lookahead 最优化方法;场景文本识别 | 交通量预测:<br />[https://momodel.cn/live/5d4b96981afd943042ef1603](https://momodel.cn/live/5d4b96981afd943042ef1603)<br />Lookahead 最优化方法:<br />[https://momodel.cn/live/5d4cd97c1afd943060d021aa](https://momodel.cn/live/5d4cd97c1afd943060d021aa)<br />场景文本识别:<br />[https://momodel.cn/live/5d4bec421afd9430a4627a77](https://momodel.cn/live/5d4bec421afd9430a4627a77)<br />（电脑端打开链接，点击“进入课程”，点击“回放”按钮观看回放） |
| 20 | 8.24 | 【论文分享】做梦神经网络；联邦学习；Seq2Seq模型中的Attention机制 | 做梦神经网络：[https://momodel.cn/live/5d6064a81afd9432607db06c](https://momodel.cn/live/5d6064a81afd9432607db06c)<br />联邦学习：<br />[https://momodel.cn/live/5d600a151afd9432607db053](https://momodel.cn/live/5d600a151afd9432607db053)<br />Seq2Seq模型中的Attention机制：<br />[https://momodel.cn/live/5d5ffc891afd94183b29fc9a](https://momodel.cn/live/5d5ffc891afd94183b29fc9a) |
| 21 | 9.7 | 【论文分享】机器学习算法的预测价值；深度强化学习在机械臂接触操纵任务中的应用；基于耦合网络的推荐系统 | 直播回放：<br />[https://v.douyu.com/show/yVmjvBZldzrvqkNb](https://v.douyu.com/show/yVmjvBZldzrvqkNb) |
| 22 | 9.21 | 【论文分享】动态时间规整算法；混合联邦蒸馏；深度协作学习 | 动态时间规整：https://momodel.cn/live/5d858c582e62cf2ca9c22088 <br />  混合联邦蒸馏：https://momodel.cn/live/5d858bf5e4ed67738f25516b <br />深度协作学习:  https://momodel.cn/live/5d85be2df7ab16cba041dba3 |
| 23 | 10.19 | 【论文分享】深度强化学习路在何方；无数据的蒸馏学习；蒙特卡洛树搜索(MCTS)算法 | [深度强化学习路在何方 视频回放](https://v.douyu.com/show/yVY8WwZAADoMLOz9) <br />  [无数据的蒸馏学习 视频回放](https://v.douyu.com/show/NbwE7ZG111BMn5Zz) <br /> [蒙特卡洛树搜索 视频回放](https://v.douyu.com/show/Qyz171mQQgJ7BJj9)  <br /> [蒙特卡洛树搜索 项目实现](https://momodel.cn/explore/5daa6b431828bde9b6d740d1?type=app) |
| 24 | 11.02 | 【论文分享】基于蒸馏的联邦学习；基于主动学习的联邦学习；利用匹配分解采样策略优化分散SGD |[基于蒸馏、主动学习的联邦学习](https://momodel.cn/live/5dbd165e759ebd058aec0311)  <br /> [利用匹配分解采样策略优化分散SGD](https://momodel.cn/live/5dbd1785dece81ddeffda809)|
| 25 | 11.16 | 【论文分享】分布式优化与异构数据联邦学习算法；异步联合更新；使用局部/全局代表进行联邦学习 |[分布式优化与异构数据联邦学习算法](https://momodel.cn/live/5dce9df2e34e04fbb610de98) <br /> [联邦学习通信优化与异步联邦优化](https://momodel.cn/live/5dcecf2223b1c9191221b21e)<br />（电脑端打开链接，点击“进入课程”，点击“回放”按钮观看回放） |
|26|11.23| 【论文分享】联邦学习中数据集对训练的影响；联邦学习中的量化方法 | [联邦学习中数据集对训练的影响](https://momodel.cn/live/5dd915cafe279e6e6a024ea2) <br />[联邦学习中的量化方法](https://momodel.cn/live/5dd91aca137547ccb2f648bf)<br /> （电脑端打开链接，点击“进入课程”，点击“回放”按钮观看回放） |
|27|11.30| 【论文分享】联邦学习中的贡献度量；规模化的联邦学习系统设计 | [课程资料](https://github.com/momodel/AIClub/tree/master/SharedPaper/20191130)|
|28|12.05| 【论文分享】联邦学习背景下的MAML与域适应方法；联邦学习框架—fate | [联邦学习背景下的MAML与域适应方法](https://v.douyu.com/show/yVmjvB450NAMqkNb) <br /> [联邦学习框架—fate](https://v.douyu.com/show/jwzOvpJ3lODMZVRm)<br /> |




## 关于我们
[Mo](https://momodel.cn)（网址：https://momodel.cn） 是一个支持 Python的人工智能在线建模平台，能帮助你快速开发、训练并部署模型。

近期 [Mo](https://momodel.cn) 也在持续进行机器学习相关的入门课程和论文分享活动，欢迎大家关注我们的公众号获取最新资讯！

![](https://imgbed.momodel.cn/联系人.png)
