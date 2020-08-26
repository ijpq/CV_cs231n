import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
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
        dW[:,j]+=X[i].T
        dW[:,y[i]]+=-X[i].T
        
        
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW/=num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW+=reg*2*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores=np.dot(X,W)
  #找真实标记也用的是scores矩阵
  correct_class_score=scores[range(num_train),list(y)].reshape(-1,1)
  sum=scores-correct_class_score+1#计算relu中第二项
  sum[range(num_train),list(y)]=0#去掉j=y[i]
  margins=np.maximum(0, sum)
  loss=np.sum(margins)/num_train +  reg* np.sum(W*W)#计算relu,求和得到loss

  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  


  coeff_mat = np.zeros((num_train, num_classes))
  coeff_mat[margins > 0] = 1
  coeff_mat[range(num_train), list(y)] = 0
  coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)

  dW = (X.T).dot(coeff_mat)
  dW = dW/num_train + reg*W*2
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
