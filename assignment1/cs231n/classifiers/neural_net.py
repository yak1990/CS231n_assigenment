from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        a=np.matmul(X,W1)+b1
        now_mask1=a>0
        now_mask2=a<=0
        a[now_mask2]=0
        b=a
        c=np.matmul(b,W2)+b2
        d=np.exp(c)
        e=np.sum(d,axis=1)
        f=(d.T/e).T
        scores=c

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss=0
        now_mask=np.zeros(f.shape)
        for i,j in enumerate(y):
          now_mask[i,j]=1
        loss=np.sum(-np.log(f)*now_mask)
        loss/=X.shape[0]
        loss+=reg*np.sum(W1*W1)
        loss+=reg*np.sum(W2*W2)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        grads['W2']=np.matmul(b.T,f)+np.matmul(b.T,(-now_mask))
        grads['W2']/=X.shape[0]
        grads['W2']+=2.0*reg*W2

        now_b=np.ones(X.shape[0])
        grads['b2']=np.matmul(now_b,f)+np.matmul(now_b,(-now_mask))
        grads['b2']/=X.shape[0]

        tmp_a=(np.matmul(W2,f.T)+np.matmul(W2,(-now_mask).T)).T
        tmp_a[now_mask2]=0
        tmp_b=tmp_a
        tmp_c=np.matmul(X.T,tmp_b)
        grads['W1']=tmp_c/X.shape[0]
        grads['W1']+=2.0*reg*W1
        
        grads['b1']=np.matmul(now_b,tmp_b)
        grads['b1']/=X.shape[0]

        '''
        def compute_loss():
          a=np.matmul(X,W1)+b1
          now_mask1=a>0

          now_mask2=a<=0
          a[now_mask2]=0
          b=a

          c=np.matmul(b,W2)+b2
          d=np.exp(c)
          e=np.sum(d,axis=1)
          f=(d.T/e).T
          scores=f
          out = 0
          now_mask=np.zeros(scores.shape)
          for i,j in enumerate(y):
            now_mask[i,j]=1
          out=np.sum(-np.log(scores)*now_mask)
          out/=X.shape[0]
          out+=reg*np.sum(W1*W1)
          out+=reg*np.sum(W2*W2)
          return out

        def compute_diff_W1(delta):
          dw=np.zeros(W1.shape)
          for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
              W1[i,j]+=delta
              now_loss=compute_loss()
              dw[i,j]=(now_loss-loss)/delta
              W1[i,j]-=delta
          print("diff_W1:",dw)

        
        def compute_diff_W2(delta):
          dw=np.zeros(W2.shape)
          for i in range(W2.shape[0]):
            for j in range(W2.shape[1]):
              W2[i,j]+=delta
              now_loss=compute_loss()
              dw[i,j]=(now_loss-loss)/delta
              W2[i,j]-=delta
          print("diff_W2:",dw)
        
        def compute_diff_b1(delta):
          dw=np.zeros(b1.shape)
          for i in range(b1.shape[0]):
              b1[i]+=delta
              now_loss=compute_loss()
              dw[i]=(now_loss-loss)/delta
              b1[i]-=delta
          print("diff_b1:",dw)

        
        def compute_diff_b2(delta):
          dw=np.zeros(b2.shape)
          for i in range(b2.shape[0]):
              b2[i]+=delta
              now_loss=compute_loss()
              dw[i]=(now_loss-loss)/delta
              b2[i]-=delta
          print("diff_b2:",dw)
        
        print("now_loss:",loss,"\ncompute_loss:",compute_loss())
        delta=1e-6
        compute_diff_W1(delta)
        compute_diff_W2(delta)
        compute_diff_b1(delta)
        compute_diff_b2(delta)
        print("def_w1",grads['W1'])
        print("def_w2",grads['W2'])
        
        print("def_b1",grads['b1'])
        print("def_b2",grads['b2'])
        '''

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            if(num_train>batch_size):
              batch_mask=np.random.choice(num_train,batch_size,replace=False)
              X_batch=X[batch_mask]
              y_batch=y[batch_mask]
            else:
              X_batch=X
              y_batch=y

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            self.params['W1']-=learning_rate*grads['W1']
            self.params['W2']-=learning_rate*grads['W2']
            self.params['b1']-=learning_rate*grads['b1']
            self.params['b2']-=learning_rate*grads['b2']

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        y_pred = np.zeros(X.shape[0])
        a=np.matmul(X,W1)+b1
        now_mask1=a>0
        now_mask2=a<=0
        a[now_mask2]=0
        b=a
        c=np.matmul(b,W2)+b2
        for i in range(X.shape[0]):
          y_pred[i]=np.argmax(c[i,:])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred

def my_test():
    data_size=10
    dim=3
    pred_size=5
    X=np.random.rand(data_size,dim)*1e7
    W=np.random.rand(dim,pred_size)
    y=np.random.randint(low=0,high=5,size=data_size)
    
    tmp=TwoLayerNet(dim,4,pred_size)
    tmp.loss(X,y,0.2)
    print(tmp.predict(X))



if __name__ =="__main__":
    for i in range(10):
        my_test()
        print("\n\n\n")