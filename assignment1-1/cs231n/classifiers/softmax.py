import numpy as np
from random import shuffle
from past.builtins import xrange

import math

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train=X.shape[0]
  num_classes=W.shape[1]
  dim=X.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores=np.dot(X,W)
  #NOTE:log是ln
  '''
  提高数值稳定性计算法
  '''

  for i in xrange(num_train): 
    numerator=0
    denominator=0
    const_stability = -1 * np.amax(scores[i])#数值稳定性 LOGC
    sum = 0
    numerator = math.exp(scores[i][y[i]]+const_stability)#每一个样本i有一个分子
    for j in xrange(num_classes):  #计算分母,每一个样本i有一个分母
      denominator += math.exp(scores[i][j]+const_stability)
      # sum += math.exp(scores[i][j]+const_stability)
    for j in xrange (num_classes):#更新梯度
      if j==y[i]:
        dW[:, j] += (((math.exp(scores[i][j]+const_stability)) /
                      denominator)-1)*X[i].T
                      #X[i]转不转置都是D维的.
        pass
      elif j!=y[i]:
        dW[:, j] += (math.exp(scores[i][j]+const_stability)
                     * X[i].T) / denominator
        pass 
    
    loss+= -1 * math.log(numerator/denominator)
  loss/=num_train
  loss += reg* np.sum(W**2)

  dW/=num_train
  dW+= 2*reg*W
  

  

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  num_train=X.shape[0]
  num_classes=W.shape[1]
  loss = 0.0
  dW = np.zeros_like(W)
  

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W)
  shift_scores = scores-(np.amax(scores, axis=1).reshape(-1, 1))
  shift_scores= np.exp(shift_scores)
  
  #gradient
  denominator=np.sum(shift_scores,axis=1).reshape(-1,1)
  A=shift_scores/denominator
  A[range(num_train),y]-=1
  dW = np.dot(X.T, A)/num_train + 2 * reg * W


  #checked
  loss=np.sum(np.log(np.sum(np.exp(shift_scores),axis=1))-shift_scores[range(num_train),y])
  loss/=num_train 
  loss += reg* np.sum(W**2)




  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

