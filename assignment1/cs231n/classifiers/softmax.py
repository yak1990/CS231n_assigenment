from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def compute_loss():
      out=0
      for i in range(X.shape[0]):
        score=X[i,:].dot(W)
        score=np.exp(score)
        sum_score=sum(score)
        out+=-np.log(score[y[i]]/sum_score)
      out/=X.shape[0]
      out+=reg*np.sum(W*W)
      return out

    
    for i in range(X.shape[0]):

      score=X[i,:].dot(W).reshape(W.shape[1],1)
      score=np.exp(score)
      sum_score=sum(score)
      loss+=-np.log(score[y[i]]/sum_score)
      score/=sum_score
      now_dw=X[i,:].T*score
      now_dw[y[i],:]-=X[i,:]
      dW+=now_dw.T
    loss/=X.shape[0]
    dW/=X.shape[0]
    loss+=reg*np.sum(W*W)
    dW+=2.0*reg*W

    '''
    delta=0.001
    ori_loss=compute_loss()
    diff_dW=np.zeros(dW.shape)
    for i in range(W.shape[0]):
      for j in range(W.shape[1]):
        W[i,j]+=delta
        now_loss=compute_loss()
        diff_dW[i,j]=(now_loss-ori_loss)/delta
        W[i,j]-=delta

    print("diff:",diff_dW)
    print("der:",dW)
    '''
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train=X.shape[0]
    score=np.matmul(X,W)
    score=np.exp(score)
    sum_score=np.sum(score,axis=1)
    score=(score.T/sum_score).T
    #score/=sum_score
    mask=np.zeros(score.shape)
    for i,j in enumerate(y):
      mask[i,j]=1
    loss=np.sum(-np.log(score)*mask)
    loss/=X.shape[0]
    loss+=reg*np.sum(W*W)
    
    
    dW=np.matmul(X.T,score)+np.matmul(X.T,(-mask))
    dW/=X.shape[0]
    dW+=2.0*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

def my_test():
    data_size=10
    dim=3
    pred_size=5
    X=np.random.rand(data_size,dim)
    W=np.random.rand(dim,pred_size)
    y=np.random.randint(low=0,high=5,size=data_size)
    #print(y)
    #print(X.shape,W.shape,y.shape)
    loss,grad=softmax_loss_naive(W,X,y,0.5)
    print(loss)
    print(grad)

    loss,grad=softmax_loss_vectorized(W,X,y,0.5)
    print(loss)
    print(grad)


if __name__ =="__main__":
    for i in range(1):
        my_test()
        print("\n\n\n")