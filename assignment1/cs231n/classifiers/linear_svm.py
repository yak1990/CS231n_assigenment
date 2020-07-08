from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive_ori(W, X, y, reg):
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
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW

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
    def computeLoss(now_w):
        out_loss=0.0
        for i in range(num_train):
            scores = X[i].dot(now_w)
            correct_class_score = scores[y[i]]
            for j in range(num_classes):
                if j == y[i]:
                    continue
                margin = scores[j] - correct_class_score + 1 # note delta = 1
                if margin > 0:
                    out_loss += margin

        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        out_loss /= num_train

        # Add regularization to the loss.
        out_loss += reg * np.sum(W * W)
        return out_loss
    loss=computeLoss(W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def compute_dif(delta):
        dW = np.zeros(W.shape)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
            #print(i,j)
                W[i][j]+=delta
                now_loss=computeLoss(W)
                dW[i][j]=(now_loss-loss)/delta
                W[i][j]-=delta
        print("dif:",delta,"\n",dW)
    #compute_dif(0.1)
    #compute_dif(0.01)
    #compute_dif(0.001)
    dW = np.zeros(W.shape)
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                dW[:,j]+=X[i]
                dW[:,y[i]]-=X[i]
    dW/=num_train
    dW+=2.0*reg*W
    
    #print("dif:\n",diff_dw)
    #print("der:\n",dW)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #corr_matrix=np.zeros(all_loss.shape)
    all_loss=np.matmul(X,W)
    
    corr_score=np.array([all_loss[i][j] for i,j in zip(range(X.shape[0]),y)]).reshape(X.shape[0],1)
   
    all_loss=all_loss-corr_score
    all_loss=all_loss+1
    mask=all_loss>0
    loss=np.sum(all_loss[mask])-X.shape[0]
    loss /= X.shape[0]
    loss += reg * np.sum(W * W)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    count=np.sum(mask==True,axis=1)
    #print(count.shape)
    #print(count)
    A=np.zeros((X.shape[0],W.shape[1]))
    #print(A.shape)
    #print(mask.shape)
    A[mask]=1
    for i,j in enumerate(y):
        A[i,j]=-count[i]+1
    #print("A:\n",A)
    dW=np.matmul(X.T,A)
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
    loss,grad=svm_loss_naive(W,X,y,0.5)
    print(loss)
    print(grad)

    loss,grad=svm_loss_vectorized(W,X,y,0.5)
    print(loss)
    print(grad)


if __name__ =="__main__":
    for i in range(10):
        my_test()
        print("\n\n\n")
