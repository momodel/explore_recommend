
<h3 align="center" color='blue' class='test'> 第一回合：春 </h3>

<p align="center"> 《春晓》孟浩然 </p>

<div class='textWrap'>
  <div>春眠不觉晓，处处闻啼鸟。 </div>
  <div>夜来风雨声，花落知多少。 </div>
</div>


<p align="center"> 《春眠啼鸟》Mo </p>
<div class='textWrap'>
 <div><span class='poetry'>春</span>去两三杯，</div>
 <div><span class='poetry'>眠</span>宵远樵者。</div>
 <div><span class='poetry'>啼</span>风容发初，</div>
 <div><span class='poetry'>鸟</span>雀鸣山色。</div>
</div>





<p align="center"> 第二回合：夏 </p>
<p align="center"> 《晓出净慈寺送林子方》杨万里  </p>

<div class='textWrap'>
  <div>毕竟西湖六月中，风光不与四时同。</div>
  <div>接天莲叶无穷碧，映日荷花别样红。 </>
</div>



<p align="center"> 《映日荷花》Mo     </p>
<div class='textWrap'>
 <div><span class='poetry'>映</span>真林下逢来会，</div>
 <div><span class='poetry'>日</span>日宁令物俗情。</div>
 <div><span class='poetry'>荷</span>馆浓香春欲尽，</div>
 <div><span class='poetry'>花</span>开花落月明新。</div>
</div>



    
<p align="center"> 第三回合：秋 </p>

<p align="center"> 《山居秋暝》王维 </p>
<div class='textWrap'>
  <div>空山新雨后，天气晚来秋。</div>
  <div>明月松间照，清泉石上流。</>
  <div>竹喧归浣女，莲动下渔舟。</div>
  <div>随意春芳歇，王孙自可留。</div>
</div>



<p align="center" > 《空山新雨》Mo </p>
<div class='textWrap'>
 <div><span class='poetry'>空</span>窗增达趣，</div>
 <div><span class='poetry'>山</span>静步萝丛。 </div>
 <div><span class='poetry'>新</span>幄寻池上，</div>
 <div><span class='poetry'>雨</span>声清夜钟。</div>
</div>





<p align="center"> 第四回合：冬 </p>
<p align="center"> 《江雪》柳宗元  </p>
<div class='textWrap'>
 <div>千山鸟飞绝，万径人踪灭。 </div>
 <div>孤舟蓑笠翁，独钓寒江雪。</div>
</div>



<p align="center"> 《千山江雪》Mo </p>

<div class='textWrap'>
  <div><span class='poetry'>雨</span>里青台头粉掌，</div>
  <div><span class='poetry'>山</span>光青黛落如烟。</div>
  <div><span class='poetry'>江</span>月最烧墙数望，</div>
  <div><span class='poetry'>雪</span>中无事不相看。</div>
</div>




怎么样，AI 没有让你失望吧？Mo 写出的诗句不仅和四季相关，还从“对手”的诗句中提取了关键词进行藏头，这波666的操作想不佩服都不难！下面让我们一起来了解 AI 写诗背后的奥秘——深度学习算法。

### 深度学习
深度学习是一类机器学习方法，可实例化为深度学习器，所对应的设计、训练和使用方法集合称为深度学习。深度学习器由若干处理层组成，每层包含至少一个处理单元，每层输出为数据的一种表征，且表征层次随处理层次增加而提高。

深度的定义是相对的。针对某具体场景和学习任务，若学习器的处理单元总数和层数分别为M和N，学习器所保留的信息量或任务性能超过任意层数小于N且单元总数为M的学习器，则该学习器为严格的或狭义的深度学习器，其对应的设计、训练和使用方法集合为严格的或狭义的深度学习。

深度学习听起来高深，落地的应用却可以很浪漫。比如作诗、作曲、人脸美容美妆等都可以实现。下面我们以古诗词生成器为例，一步一步带你从数据处理到模型搭建，再到训练出古诗词生成模型

### LSTM 介绍
像诗词文本这样的数据，文字的前后文存在关联性被称为序列化数据，即前一数据和后一个数据有顺序关系。深度学习中有一个重要的分支是专门用来处理这样的数据的——循环神经网络。循环神经网络广泛应用在自然语言处理领域(NLP)，今天我们带你介绍循环神经网络一个重要的改进算法模型-LSTM。这里不对LSTM的原理进行深入，想要深入理解LSTM的可以戳这里[《[译] 理解 LSTM 网络》](https://www.jianshu.com/p/9dc9f41f0b29)。


### 数据处理

我们使用76748首古诗词作为数据集，数据集[下载链接](http://www.momodel.cn:8899/#/explore/5c00a6e21afd942b66b36ba8?type=dataset)，原始的古诗词的存储形式如下：
![image](https://user-images.githubusercontent.com/43362551/51824023-221ea180-231c-11e9-8577-6595844d752f.png)
我们可以看到原始的古诗词是文本符号的形式，无法直接进行机器学习，所以我们第一步需要把文本信息转换为数据形式，这种转换方式就叫词嵌入(word embedding)，我们采用一种常用的词嵌套(word embedding)算法-Word2vec对古诗词进行编码。关于Word2Vec这里不详细讲解，有兴趣的可以参考[《[NLP] 秒懂词向量Word2vec的本质》](https://zhuanlan.zhihu.com/p/26306795)。在词嵌套过程中，为了避免最终的分类数过于庞大，可以选择去掉出现频率较小的字，比如可以去掉只出现过一次的字。Word2vec算法经过训练后会产生一个模型文件，我们就可以利用这个模型文件对古诗词文本进行词嵌套编码。

经过第一步的处理已经把古诗词词语转换为可以机器学习建模的数字形式，因为我们采用LSTM算法进行古诗词生成，所以还需要构建输入到输出的映射处理。例如：
“[长河落日圆]”作为train_data，而相应的train_label就是“长河落日圆]]”，也就是
“[”->“长”，“长”->“河”，“河”->“落”，“落”->“日”，“日”->“圆”，“圆”->“]”，“]”->“]”，这样子先后顺序一一对相。这也是循环神经网络的一个重要的特征。
这里的“[”和“]”是开始符和结束符，用于生成古诗的开始与结束标记。

总结一下数据处理的步骤：
- 读取原始的古诗词文本，统计出所有不同的字，使用 Word2Vec 算法进行对应编码；
- 对于每首诗，将每个字、标点都转换为字典中对应的编号，构成神经网络的输入数据 train_data；
- 将输入数据左移动构成输出标签 train_label；

经过数据处理后我们得到以下数据文件： 
- poems_edge_split.txt：原始古诗词文件，按行排列，每行为一首诗词；
- vectors_poem.bin：利用 Word2Vec训练好的词向量模型，以</s>开头，按词频排列，去除低频词；
- poem_ids.txt：按输入输出关系映射处理之后的语料库文件；
- rhyme_words.txt： 押韵词存储，用于押韵诗的生成；


### 模型构建及训练
这里我们使用2层的LSTM框架，每层有128个隐藏层节点，我们使用tensorflow.nn模块库来定义网络结构层，其中RNNcell是tensorflow中实现RNN的基本单元，是一个抽象类，在实际应用中多用RNNcell的实现子类BasicRNNCell或者BasicLSTMCell，BasicGRUCell；如果需要构建多层的RNN，在TensorFlow中，可以使用tf.nn.rnn_cell.MultiRNNCell函数对RNNCell进行堆叠。模型网络的第一层要对输入数据进行 embedding，可以理解为数据的维度变换，经过两层LSTM后，接着softMax得到一个在全字典上的输出概率。
模型网络结构如下：
![image](https://user-images.githubusercontent.com/43362551/51891576-8142eb80-23da-11e9-84c4-66ffdf971818.png)

训练时可以定义batch_size的值，是否进行dropout，为了结果的多样性，训练时在softmax输出层每次可以选择topK概率的字符作为输出。训练完成后可以使用tensorboard 对网络结构和训练过程可视化展示。这里推荐用咱们自家建模平台Mo，带有完整的Python和机器学习框架运行环境，并且有免费的GPU可以使用，大家可以自己试试哦。

### 诗词生成
调用前面训练好的模型我们就可以实现一个古诗词的应用了，刚刚 Mo 写的诗就是这样生成的：
![春](https://ws2.sinaimg.cn/large/006tKfTcly1g0vf6edygjj318a0isdha.jpg)
![夏](https://ws4.sinaimg.cn/large/006tKfTcly1g0vf7zrbx3j318a0hgjt7.jpg)
![秋](https://ws4.sinaimg.cn/large/006tKfTcly1g0vf8k3fwgj318e0immyp.jpg)
![冬](https://ws2.sinaimg.cn/large/006tKfTcly1g0vf8y7ukhj318k0iqwgc.jpg)

