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
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,y[i]] -= X[i] # cost func derived by Wyi = -Xi
        dW[:,j] += X[i] # cost function derived by Wj = Xi


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores_over_all_sample = X.dot(W)
  # https://docs.scipy.org/doc/numpy/user/basics.indexing.html
  correct_class_score_over_all_sample = scores_over_all_sample[np.arange(y.shape[0]), y]
  # transpose of correct_class_score_over_all_sample
  correct_class_score_over_all_sample_T = correct_class_score_over_all_sample.reshape(-1, 1)
  # calculate margin
  margin_over_all_sample = scores_over_all_sample - correct_class_score_over_all_sample_T + 1
  margin_over_all_sample_positive = np.maximum(margin_over_all_sample, 0)
  loss = np.sum(margin_over_all_sample_positive)/y.shape[0] - 1
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
  mask = np.zeros(margin_over_all_sample_positive.shape)
  # all positive margin except for the right class would be plus Xi
  mask[margin_over_all_sample_positive > 0] = 1
  # all right-class margin would be -Xi * (num of positive margins)
  # for the +1 here: margin for the expected class is be calculated
  # as positive margin 1 which should be zero
  mask[np.arange(y.shape[0]), y] = -np.sum(mask, axis=1) + 1
  dW = X.T.dot(mask)/y.shape[0]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
