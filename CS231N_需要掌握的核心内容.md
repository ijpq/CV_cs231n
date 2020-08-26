[TOC]

```markdown
说明：记录的是用自己的话对课程以及作业进行得理解，顺序上可能存在出入，回来看的时候如果发现有不理解或者不对的地方查阅
[大神发布在知乎上的 CS231N 笔记]: https://zhuanlan.zhihu.com/p/21930884
```



***

#  Module 1: 神经网络

## [Linear classification: Support Vector Machine, Softmax](http://cs231n.github.io/linear-classify/)
### Loss function

#### Multiclass SVM loss

第j个类别的评分是第j个元素 $s_j = f(x_i, W)_j$,其中$f(x_i, W) =  W x_i$

![006tKfTcly1g10uzwmggtj31130d8abg](assets/006tKfTcly1g10uzwmggtj31130d8abg-3331027.jpg)

The Multiclass SVM loss for the i-th example is then formalized as follows:

$L_i = \sum_{j\neq y_i} \max(0, s_j - s_{y_i} + \Delta)$

假如有三个类,然后得到评分为13,-7,11.其中第一个类是真实标记.并且假设delta=10.

##### Regularization

多类别SVM损失函数展开后如下:

$L = \frac{1}{N} \sum_i \sum_{j\neq y_i} \left[ \max(0, f(x_i; W)_j - f(x_i; W)_{y_i} + \Delta) \right] + \lambda \sum_k\sum_l W_{k,l}^2$ (公式1)

使用L2范数做正则化时会得到SVM中很不错的margin值

最好的性质是惩罚大的权重可以提高泛化能力,因为它意味着输入的维度都只产生较小的影响.举个例子,假定有两个权重向量[1,0,0,0]和[0.25,0.25,0.25,0.25],这两个权重乘以$x = [1,1,1,1]$都等于1.所以两个权重向量产生了相同的点积结果.但是L2范数惩罚对于第一个向量计算出来是1,对于第二个向量是0.25.因此根据L2范数惩罚,第二个向量具有更小的正则化损失,泛化性能更好也是指样本xi对第一个维度的依赖程度相对更低.

### Practical Considerations

#### setting delta

delta 和 正则项的因子都是平衡数据损失与正则损失之间的大小.理解的关键是权重矩阵的大小对最后类别评分之间的差值有直接影响:如果我们把权重矩阵的值整体调小,那么各类别评分之间的差值也会缩小.因此,分数之间的差值在某种程度上没什么意义,因为权重矩阵可以任意的整体放大或缩小.真正有意义的是正则项因子的大小调整.

#### 和二分类SVM的关系

二分类的损失函数:$L_i = C \max(0, 1 - y_i w^Tx_i) + R(W)$是*公式1*的特例,

超参数C与1/$\lambda$成正比

#### 优化

#### 其他多分类SVM表达式

### Softmax classifier

<span style="color:red">softmax中的log是ln</span>

<span style="color:red">这块的理解可能有误，需要修改。TODO:</span>

https://blog.csdn.net/u014380165/article/details/77284921

---

softmax classifier 是 logistic regression多分类 classifier 的一般形式.

不像SVM分类器,把输出$f(x_i,W)$当做是当前样本在每个类的评分,softamax分类器把输出当做是一种概率的理解.在Softmax中，$f(x_i; W) =  W x_i$不变,但hingeloss替换成cross-entropy loss.

**cross-entropy loss** : $L_i = -\log\left(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }\right) \hspace{0.5in} \text{or equivalently} \hspace{0.5in} L_i = -f_{y_i} + \log\sum_j e^{f_j}$

交叉熵loss中fj是类别评分向量f的第j个元素=$(Wx_{i})_j$

像以前一样,full loss 是数据集上($\sum$Li + 正则项) 的均值

**softmax function** : $f_j(z) = \frac{e^{z_j}}{\sum_k e^{z_k}}$,softmax函数把向量的值挤压成一个0到1之间值得向量,并且求和等于1.

SVM classifier 使用 hinge loss ,也被称为max-margin loss.

softmax classifier 使用cross-entropy loss,这个分类器的得名是因为使用了softmax函数.由于softmax只是个把数值压到01之间的函数,所以不要说softmax loss.

#### Practical issues: Numeric stability

写代码计算softmax函数时,中间变量$e^{f_{y_i}}$和$\sum_j e^{f_j}$在指数e的作用下可能会非常大.除以一个很大的数会导致数值不稳定,所以在这里使用一个归一化的小技巧很必要.注意到,如果分子分母同时乘以一个常数C,并把常数C放到指数上去,可得到如下表达式:

$\frac{e^{f_{y_i}}}{\sum_j e^{f_j}} = \frac{Ce^{f_{y_i}}}{C\sum_j e^{f_j}} = \frac{e^{f_{y_i} + \log C}}{\sum_j e^{f_j + \log C}}$

计算结果不会变,但是可以提高数值稳定性.常数C的普遍选择是$\log C = -\max_j f_j$.经过这个操作之后,向量f中的最大值会变成0.

### SVM和SOFTMAX的关系

![006tKfTcly1g10e64he2aj30rx0dkq48](assets/006tKfTcly1g10e64he2aj30rx0dkq48-3331027.jpg)

两个都是计算类评分向量,不同的是对于这个向量的理解

SVM把向量的值理解为类的评分,其损失函数鼓励正确的类别的评分比错误的评分高出至少一个margin;而softmax分类器把分数理解为每个类的log概率,鼓励正确类的log概率更高.

SVM的目标是正确类的分比错误类的分高出一个margin;但是softmax目标是让错误类的loss趋于正无穷,让正确类的loss趋于0.

## [Optimization: Stochastic Gradient Descent](http://cs231n.github.io/optimization-1/)

一维梯度计算公式:$\frac{df(x)}{dx} = \lim_{h\ \to 0} \frac{f(x + h) - f(x)}{h}$

### 计算梯度

有两种方式计算梯度,一种是速度慢的,近似的但是简单的方式,数值梯度;另一种是快速的,准确的,但使用微积分且更可能出错的方式,解析梯度.

#### 用有限差分的方式来计算数值梯度

下面这个函数用来计算f在x处的梯度

根据上面给的梯度计算公式,这段代码在每个维度上增加一个小的步长h,然后在该方向上计算偏导数,并观察函数值是如何变化的.

```python
def eval_numerical_gradient(f, x):
  """ 
  a naive implementation of numerical gradient of f at x 
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """ 

  fx = f(x) # evaluate function value at original point
  grad = np.zeros(x.shape)
  h = 0.00001

  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value + h # increment by h
    fxh = f(x) # evalute f(x + h)
    x[ix] = old_value # restore to previous value (very important!)

    # compute the partial derivative
    grad[ix] = (fxh - fx) / h # the slope
    it.iternext() # step to next dimension

  return grad
```

举个例子,如下:

![006tKfTcly1g10v1r6zlxj31o00u0dqn](assets/006tKfTcly1g10v1r6zlxj31o00u0dqn-3331027.jpg)

但是以上这种计算方式非常慢,有很多缺点

#### 用微积分计算解析梯度

用有限差分近似公式可以很简单的计算出数值梯度,但缺点是这是近似的.而且计算起来很费劲.

计算解析梯度的方法可以直接得到一个精确的表达式,而且计算起来也很快.然而,解析梯度容易计算出错.有一个办法解决这个问题:**梯度check**

1. 计算出解析梯度
2. 和数值梯度比较,以确保解析梯度的正确性


用SVM loss 函数来举例:

SVM LOSS:$L_i = \sum_{j\neq y_i} \left[ \max(0, w_j^Tx_i - w_{y_i}^Tx_i + \Delta) \right]$

求偏导数:$\nabla_{w_{y_i}} L_i = - \left( \sum_{j\neq y_i} \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0) \right) x_i$

1 is the indicator function that is one if the condition inside is true or zero otherwise.

***

but the core idea of following the gradient (until we're happy with the results) will remain the same.

但是(在我们对结果感到满意之前)遵循渐变的核心思想将保持不变。

*not…until…* : until的条件或时间达不到,前面就一直not

***

### 梯度下降

#### 小批量梯度下降

由于数据量巨大,所以根据所有的训练样本计算loss的一个参数太亏了.因而采取一个batch的训练数据来计算.举个例子,120万的训练数据中一个batch包含256个样本.

这种方法好用的原因是,训练数据中的样本是相互关联的.考虑一个极端的例子,120万的数据集只是由1000个图片的副本复制出来的(120万/0.1万=1200.故对于每一个原本,有1200张副本).显然对1200个副本计算梯度是一样的.当我们对120万训练数据计算平均loss的时候,与对1000张图像计算平均loss是一样的.在实际操作中,数据集不会有重复的样本,根据小批量计算的梯度是对整个训练数据梯度的很好的近似.因此用小批量梯度下降可以得到更快的收敛而且参数更新的也越快.

#### 随机梯度下降SGD

小批量中一个batch中只包含一个样本,就是随机梯度下降.

这个不怎么用了,因为计算一个100样本的batch比计算1个样本100次要快.3

### 更进一步的优化

#### SGD+MOMENTUM

![Screenshot 2019-06-26 15.11.03](assets/Screenshot 2019-06-26 15.11.03.png)第一次时，vx 等于 T 度，第二次时，vx = 在当前点的 T 度上加上一个带 rho 衰减效果的上一次的 复合T 度，总之效果上等于对当前点的 T 度产生了一个拉伸方向的作用。这就使得如果在某个鞍点或局部极小值点 T 度为零时，可以在上一次复合 T 度的帮助下，使当前复合 T 度得到一个不等于零的结果。

#### Adagrad

![Screenshot 2019-06-26 15.20.31](assets/Screenshot 2019-06-26 15.20.31.png)

每次更新参数的时候，不仅要用学习率对 T 度进行衰减，还要各维度上除以 T 度的平方，这就使得：如果像图中 x 轴方向 T 度很大，那么该维度除以对应维度上参数的平方，则会使该维度参数变化减慢，而 y 轴对应的维度效果反之，总体上来看就平衡了各维度之间不协调的更新速度。**不过这个优化方法带来的问题是：**因为 grad_squared 始终累加，使得更新的步长越来越小，这在凸函数优化时效果很好，但是非凸函数优化时容易使我们卡在局部极小值上。

#### RMSProp

![Screenshot 2019-06-26 15.32.06](assets/Screenshot 2019-06-26 15.32.06.png)

adagrad 的问题是 grad_squared 会不断累加，现在要解决这个问题。那么，第一轮就去一个非常小的T 度平方，之后每轮都叠加一个非常弱的 T 度平方，同时叠加 grad_squared 时也不是完全以系数 1 叠加，而是去一个衰弱系数，就在原来''+=''这个计算基础上进行了弱化。~~在方向上有一个对 T 度平方项的拉伸(缩小)，同时在大小上对上一次的复合 T 度（grad_squared）有一个衰减，我觉得是结合了 adagrad 方法和 momentum 方法各自的优点。~~我觉得就是把 adagrad 中步长不断累加的效应进一步弱化了。

#### Adam

![Screenshot 2019-06-26 15.41.11](assets/Screenshot 2019-06-26 15.41.11.png)

第二动量是 RMSProp 的思想，最后除以第二动量的平方根是 Adagrad 的思想，第一动量的计算是 momentum 的思想。

![新文档 2019-06-26 15.50.50_1](assets/新文档 2019-06-26 15.50.50_1-1535536.jpg)

完整的 Adam 方法还有一个无偏估计的纠正。

### Dropout

#### 为什么要在测试时对数据进行 p 的放大

因为在测试时，所有的神经元会处理全部的 input 数据，因此我们想要在测试时让神经元的输出与训练时相同。比如说，p=0.5时，测试时输出要减半才能保持和训练时有相同规模的输出。（训练时，一般有p=0.5倍的输入扔掉了，所以测试时要同样做这个尺度的放缩）

## [Backpropagation, Intuitions](http://cs231n.github.io/optimization-2/)

* 这一章如果有BP看不懂的地方就看这张图,一点一点把计算公式写出来.注意体会反向传播的时候是怎么传递的.
* 这一章教了一个很重要的方法就是**COMPUTATIONAL GRAPH**

![006tKfTcly1g10v1qn4f4j30pc0e6mxs](assets/006tKfTcly1g10v1qn4f4j30pc0e6mxs.jpg)

* "向量的T度总是和向量的尺寸一样,T度中的一个元素代表着这个元素对最终函数输出的影响程度"

![屏幕快照 2019-03-21 下午5.47.45](assets/屏幕快照 2019-03-21 下午5.47.45.png)

* 计算q关于W的T度:<span style="color:red">注意这里是不能求向量q对矩阵W的导数的</span>

* 应该求的是f(q)对W的T度,如图所示.

  向量q的分量对W的求导=对应Wij的xi分量

## [Neural Networks Part 1: Setting up the Architecture](http://cs231n.github.io/neural-networks-1/)

---

### Quick intro

在线性分类那一章中,我们使用s=Wx计算类别评分,其中假设W是$10*3072$维,x是$3072*1$维.

在如图所示的两层神经网路中,假设W1是$100*3072$维的一个把图像映射到100维中间向量的投影矩阵.函数max(0,-)是一个<span style="color:red">非线性</span>的函数.W2是$10*100$维,这个10就是每个类的评分了.我们又很多的非线性函数可以选择,但是使用relu是一个把负数变成0的普遍选择.

这个<span style="color:red">非线性</span>是重要的,如果把relu扔掉,那么W1,W2就会✖️起来变成一个单独的矩阵,这样的话我们的最终预测就会是输入的线性输出.(从$s = W_2 \max(0, W_1 x)$变成了$s=W_2*W_1x$)

W1,W2是在BP过程中使用随机梯度下降学习出来的.

![006tKfTcly1g10v1rntmjj31f40p6te1](assets/006tKfTcly1g10v1rntmjj31f40p6te1.jpg)

>1. 向量h=max(0,$W_1*x$)
>2. W1包含很多不同的模板,比如在"CAR"这一类中,W1包含了"RED CAR","YELLOW CAR"等等的模板,然后通过计算$W_1*x$可以得到一个中间向量h($100*1$)这个向量表明了,图像在各种各样的模板中每个模板的得分,比如"RED CAR"得2分,"YELLOW CAR"得3分等等.然后W2是各种模板的加权,让你可以在每个类中多个模板间权衡,来确定最后该类的评分s.
>3. 如果图像x是一个🐴的左侧脸,而W1中的模板既有🐴的左侧脸又有🐴的右侧脸,那么h向量在左侧脸的得分很高,在右侧脸的得分相对较低.但是W2不是一个求max的函数,要记住他是一个模板得分的加权函数,本例中的🐴的图像,在🐴这一类的最后得分肯定仍比其他类要高.
---

### 一个神经元的模型

#### biological motivation and connections

![neuron](assets/neuron.png)

![neuron_model](assets/neuron_model.jpeg)



一个神经元前向传播的代码:

```python
class Neuron(object):
  # ... 
  def forward(self, inputs):
    """ assume inputs and weights are 1-D numpy arrays and bias is a number """
    cell_body_sum = np.sum(inputs * self.weights) + self.bias
    firing_rate = 1.0 / (1.0 + math.exp(-cell_body_sum)) # sigmoid activation function
    return firing_rate
```

每个神经元将输入数据和权重进行内积,加上偏置项再套上非线性函数(或激活函数),比如说sigmoid函数.

#### 一个神经元像一个线性分类器

因此,在神经元的输出上套一个合适的损失函数,可以把一个神经元变成一个线性分类器:

* 二值softmax分类器

  $\sigma(\sum_iw_ix_i + b)$可以看做是一个类别的概率=$P(y_i = 1 \mid x_i; w)$

  在这种解释下,可以构成cross-entropy loss,优化这个目标函数就可以生成一个二值softmax分类器.

  <span style="color:red">这块是指在激活函数外层再套一个-logx,形成损失函数</span>

  因为sigmoid输出值在01之间,所以这个分类器的预测结果基于输出值是否＞0.5


* 二值SVM分类器

  神经元的输出后套上一个max hinge loss 可以训练一个二值SVM分类器

* 正则化解释

#### 普遍使用的激活函数

* **Sigmoid**
* **Tanh**
* **ReLU**
* **Leaky ReLU**
* **Maxout**
* **TLDR**

---

### 神经网络结构

#### 层级的组织结构

##### 命名惯例

N层神经网络不包括输入层;单层神经网络=没有隐藏层(输入层直接映射到输出层),有时称LOGISTIC回归和SVMs是单层神经网络的特殊情况;

##### 输出层

与其它层的神经元不同,<span style="color:red">输出层神经元并没有激活函数</span>.这是因为最后一层的输出通常是代表了类别的评分.

#### 计算样例

![neural_net2](assets/neural_net2.jpeg)

神经网络以层的结构组织起来的一个主要原因是使用矩阵操作来计算神经网络非常方便.以上图的三层神经网络为例,输入是3维向量,每一层的链接权重被保存在一个单独的矩阵中.比如说,第一个隐藏层的权重W1是(4,3)维的.这一层所有神经元的偏置项被保存在向量b1中,这是一个4维向量.W1矩阵中每一个行向量是3维的,是当前层一个神经元在输入单元的权重.所以**np.dot(W1,x)**计算了这一层所有神经元的激活值.同样的,W2是(4,4)维的,W3是4维的向量(矩阵),是最后一层的权重.

```python
# forward-pass of a 3-layer neural network:
f = lambda x: 1.0/(1.0 + np.exp(-x)) # activation function (use sigmoid)
x = np.random.randn(3, 1) # random input vector of three numbers (3x1)
h1 = f(np.dot(W1, x) + b1) # calculate first hidden layer activations (4x1)
h2 = f(np.dot(W2, h1) + b2) # calculate second hidden layer activations (4x1)
out = np.dot(W3, h2) + b3 # output neuron (1x1)
```

其中W1,W2,W3,b1,b2,b3是可以学习的参数.

注意到,变量x是包含一个batch中所有的训练数据,而不只是一个列向量.这个矩阵x的一列是是一个样本.这样的话就可以并行计算.

# 卷积神经网络

## 

# ASSIGNMENT 1#

## 1.1 KNN EXERCISE

### 计算距离时使用单层循环

```python
	"""
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      dists[i]=np.sqrt(np.sum((X[i]-self.X_train)**2,axis=1))
      pass
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists
```

![006tKfTcly1g10v1qd9hfj30u014an3k](assets/006tKfTcly1g10v1qd9hfj30u014an3k.jpg)

### 计算时使用no loop

```python
"""
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    print(np.sum(X**2, axis=1)[:, np.newaxis].shape) #500,1
    print(np.sum(self.X_train **2,    axis=1).shape)
    dists = np.sqrt(-2 * np.dot(X, self.X_train.T) + np.sum(self.X_train **2,axis=1) + np.sum(X**2, axis=1)[:, np.newaxis])
    pass
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists
```

* 我们知道$(x-y)^2=x^2+y^2-2xy$

  所以,我们如果是计算两个向量差的平方,首先可以将视角聚焦在向量中的每个元素上,对于两个向量对应位置上的元素,有如上的展开公式.

  然后在一点一点推导就可以得到no loop的计算方法.
  ![IMG_C8A402D1029E-1](assets/IMG_C8A402D1029E-1.jpeg)

* 第二点要注意的是,

```python
  np.sum(self.X_train **2,axis=1) + np.sum(X**2, axis=1)[:, np.newaxis]
```

X_train算出来是5000维的数组,shape=(5000,);而X算出来是(500,1)的向量.

这个相加的过程是:用数组中的一个数字,依次加到(500,1)的向量上,因此最终产生(5000,500)的矩阵.

### Cross-validation

交叉验证的主要过程:

![IMG_E242A30012F8-1](assets/IMG_E242A30012F8-1.jpeg)

![IMG_112F57725693-2](assets/IMG_112F57725693-2.jpeg)

---

## 1.2 SVM EXERCISE

> <span style="color:red">这一小节复习了LOSS函数是如何计算的,如果使用最原始的方法计算以及使用向量的视角加快运算速度</span>
>
> 这一小节中所提到的X[i]都表示行向量



### 计算LOSS的梯度并保存在dW矩阵中

1. 从细节出发,对于求导方式的理解
    ![IMG_0159](assets/IMG_0159.JPG)

2. 理解了细节,再从整体出发:将L进行彻底展开有如下形式

   ![IMG_0160](assets/IMG_0160.JPG)

3. 那么有如下思考过程:

   ![IMG_0161](assets/IMG_0161.JPG)

   ---

   <span style="color:red">ATTENTION</span>上面的理解有点问题,因为PPT中给出的$Wx_i$的计算结构与代码中的计算结构相反.根据代码中的结算结构,应按如下进行分析

   ![新文档 2019-03-14 17.13.23_1](CS231N_CLASSNOTE_需要掌握的核心内容.assets/新文档 2019-03-14 17.13.23_1.jpg)

   ---

   

```python
  for i in xrange(num_train):
    scores = X[i].dot(W)#X[i]是一个样本,scores是这个样本在所有类上的评分
    correct_class_score = scores[y[i]]#假设当前样本所属真实标记类别记为2,分类器给当前样本在第2类上的评分
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      #当不是当前样本的真实标记时(比如当前样本真实标记为第2类,那么对分类器给当前样本在第j=1,3,4,...类上的评分进行计算)
      #也就是计算分类器对当前样本在第2类的评分与其他1,3,4...类的评分的margin
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:#max(0,-)
        loss += margin
        #在这里更新T度
```

所以,实际上是对向量$w_j$和$w_{y_i}$进行求导.因为他们都是列向量,所以求导结果更新到列上.而且求导结果=代码中X[i].T

对于每一个i的循环内,我们既要更新所有不等于yi的j,还要更新yi对应的那个w列.

### 以向量的视角计算SVM的LOSS,填写svm_loss_vectorized

```python
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  correct_class_scores = scores[range(num_train), list(y)].reshape(-1,1) #(N, 1)
  margins = np.maximum(0, scores - correct_class_scores +1)

```

对答案的理解:

![IMG_5BB5EA80AD57-1](assets/IMG_5BB5EA80AD57-1.jpeg)

自己的进一步分析:

![新文档 2019-03-15 14.59.36_1](assets/新文档 2019-03-15 14.59.36_1.jpg)

![新文档 2019-03-15 15.00.07_1](assets/新文档 2019-03-15 15.00.07_1.jpg)

### 以向量的视角计算dW,要求比原始计算方法快

<span style="color:red">下面这个csdn要静下来多读才好理解,已经讲得很好了</span>

```python
coeff_mat = np.zeros((num_train, num_classes))#N*C (X * W)
  coeff_mat[margins > 0] = 1 
  coeff_mat[range(num_train), list(y)] = 0#不在计算范围之内
  coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)
  #对于一个样本,要计算c-1次max,即要减去c-1个yi.这个np.sum()就是在统计yi的数量

  dW = (X.T).dot(coeff_mat)
  dW = dW/num_train + 2*reg*W
```



![screencapture-blog-csdn-net-AlexXie1996-artic le-details-79184596-2019-03-18-10_35_58_页面_1](assets/screencapture-blog-csdn-net-AlexXie1996-artic le-details-79184596-2019-03-18-10_35_58_页面_1.png)

![screencapture-blog-csdn-net-AlexXie1996-article-details-79184596-2019-03-18-10_35_58_页面_2](assets/screencapture-blog-csdn-net-AlexXie1996-article-details-79184596-2019-03-18-10_35_58_页面_2.png)

![新文档 2019-03-18 16.36.39_1](assets/新文档 2019-03-18 16.36.39_1.jpg)

## 1.3 SOFTMAX EXERCISE

> 这一小节中所提到的X[i]都是行向量

总结一下**softmax_loss_vectorized**中T度的实现思路

CSDN博客中,svm_loss_vectorized的T度实现使用了更新变量的思路,我们在这里也参考这种方法

> 哦,我们给dW[j]加上一个X[i],同时给dW[y[i]]减去一个X[i]

1. Li的形式先化简,如下:

![新文档 2019-03-21 15.43.55_1](assets/新文档 2019-03-21 15.43.55_1.jpg)

2. Li求导有如下结果:

![新文档 2019-03-21 15.43.55_20190321154724_页面_2](http://ww3.sinaimg.cn/large/006tNc79ly1g4shrht8moj31po0trtax.jpg)

其中$\begin{equation}\frac{\partial f_{j}}{\partial w_{j}}\end{equation}$都是等于X[i]^t的,这就可以联想到之前SVM的更新向量了.SVM时,对于第i个样本,我们在第j列对应坐标上写入+1或0表示是否满足更新条件.在softmax中都是要更新的,所以在(i,j)坐标位置上写上前面的系数就可以了,这个系数有两点需要注意:一是j=y[i]时j≠y[i]的系数-1;二是这个系数虽然j是看着一样的,但是i不同时是会跟着变的.所以计算这个A(i,j)的值时,与ij都是有关的.

3. 构造评分矩阵

```python
scores = np.dot(X, W) #(Wxi)j
  shift_scores = scores-(np.amax(scores, axis=1).reshape(-1, 1))#+logc,数值稳定性
  shift_scores= np.exp(shift_scores)#全部取e^x
```

4. 构造分母

对每一个Li,分母是一样的,等于这一行上所有列的求和:

```python
denominator=np.sum(shift_scores,axis=1).reshape(-1,1)
```

5. 构造系数矩阵

现在要做的是,在原有shift_scores矩阵的每个元素上➗所属行对应的分母,并让j=y[i]位置上再－1

```python
  A=shift_scores/denominator
  A[range(num_train),y]-=1
```

这里涉及到一个numpy broadcast的概念,见如下例:

```python
In [8]: a
Out[8]:
array([[1],
       [2]])

In [9]: b=np.array([[1,2],[3,4]])

In [10]: b
Out[10]:
array([[1, 2],
       [3, 4]])

In [11]: b/a
Out[11]:
array([[1. , 2. ],
       [1.5, 2. ]])
```

6. 最后一步

这一步重点理解一下CSDN对于svm更新向量怎么作用那块的讲解.

总的来说就是我们让X.T和更新向量矩阵相乘,就可以实现在dW的第j列(按照更新向量的规则)加上系数(更新值)倍的X[i].T

(图片中画的是X[i],这是因为图示中已经表示成了列向量,但是代码中我们需要写**X[i].T**)

## 1.4 two_layer_net EXERCISE

1. 分析LOSS

   ![IMG_AFC9198192CE-1](assets/IMG_AFC9198192CE-1.jpeg)

2. 画computational graph

![IMG_E036997334D8-1](assets/IMG_E036997334D8-1.jpeg)

3. 分析一个样本Li对 W2求导

   ![IMG_E4D81C628B1E-1](http://ww2.sinaimg.cn/large/006tNc79ly1g4shq5l29dj30u0140k54.jpg)

在分析一个样本(Li)的时候,会出现第三行这种维度不匹配的问题,在coding 中不会遇到这个问题

4. 整合

![IMG_B8A493D137EB-1](assets/006tNc79ly1g4shpwygz4j31ac0o610k.jpg)

**到这里就自然想到将表达式分解为两个矩阵的乘积**

![IMG_E70F2A20B0C6-1](assets/006tNc79ly1g4shofm4cwj31300u012e.jpg)



因为 W2是(H,C)维的,因此这个 $max_i$和偏导数的矩阵分别应该是(H,N),(N,C)维的.

其中$max_i$组成的矩阵MAX维度=X$*$W1结果的维度=(N,D)$*$(D,H)=(N,H)

因此在MAX与偏导数矩阵乘积时,应将MAX矩阵转置.

![IMG_58C23AAABA0E-1](http://ww1.sinaimg.cn/large/006tNc79ly1g4sho8a2foj31ac0jrq71.jpg)

5. 梯度W2 反向传播(<span style="color:red">完结</span>)

![IMG_20C6540E02D1-1](http://ww1.sinaimg.cn/large/006tNc79ly1g4shnpqd4ij30mu1ma0yt.jpg)

# ASSIGNMENT 2

## batch normalization

### 方法一：计算图

![1562486203478](assets\1562486203478.png)

### backward

#### dgamma

![新文档 2019-07-07 21.38.49_1](http://ww3.sinaimg.cn/large/006tNc79ly1g4rluqvhkxj30ww0u00yi.jpg)

#### d$\sigma^2$

计算图

![新文档 2019-07-07 21.38.49_2](http://ww3.sinaimg.cn/large/006tNc79ly1g4rmepvna4j318g0o0wgl.jpg)

计算公式

![新文档 2019-07-07 21.38.49_3](http://ww1.sinaimg.cn/large/006tNc79ly1g4rmke2ajuj30u00zsqah.jpg)

d$\sigma^2$/dxi

![新文档 2019-07-07 21.38.49_4](http://ww4.sinaimg.cn/large/006tNc79ly1g4rmomwr6aj31g40go402.jpg)

d$\mu$

![新文档 2019-07-07 21.38.49_5](http://ww3.sinaimg.cn/large/006tNc79ly1g4rmw8rfigj31r40m8juh.jpg)

dxi_hat/d$\mu$

![新文档 2019-07-06 17.24.33_1](http://ww3.sinaimg.cn/large/006tNc79ly1g4rmxmds9hj30u012j42r.jpg)

**求 d$\sigma^2$/d$\mu$时要注意紧盯计算图，注意到 xi 是常量**

![新文档 2019-07-07 21.38.49_6](http://ww3.sinaimg.cn/large/006tNc79ly1g4rn2f59rzj31wg0o00vu.jpg)

### 方法二：基于计算图的公式计算法

后面layer normalization时用这个计算公式会更方便

![1562918976937](assets/1562918976937.png)

### 遇到的问题

1. xi_hat对xi的求导计算出错

   答案应该是:标准差分之一.而我在计算时总想着xi_hat含有均值,方差,而这两个量中又包含xi,因此变成了一个复合求导.实际上要注意计算图中的三个xi是有区别的,对xi的求导计算式如下:

   ![1562486321530](assets\1562486321530.png)

   xi_hat对xi求导时,就将方差和均值看成常量**看成常量这句话的含义是要看成中间变量,而不是用再去剖析中间变量的细节,因为中间变量的细节已经放到另一项去考虑**

# 第十三章 生成式模型



![屏幕快照 2019-05-27 下午9.51.06](http://ww2.sinaimg.cn/large/006tNc79ly1g4rlgeqcs6j31c00u0hdu.jpg)

这个方法的好处是可以用一些更好的feature 来初始化我的模型