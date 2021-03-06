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
  num_of_samples = X.shape[0]
  num_of_class = W.shape[1]
  for i in range(num_of_samples):
    x_i = X[i]
    y_i = y[i]
    scores = x_i.T.dot(W)
    scores -= np.max(scores)
    exp_scores = np.exp(scores)
    probs = exp_scores/np.sum(exp_scores)
    loss += -np.log(probs[y_i])
    for j in range(num_of_class):
      dW[:, j] += x_i * (probs[j] - (1 if y_i == j else 0))
  loss = loss/num_of_samples + 0.5 * reg * np.sum(W * W)
  dW = dW/num_of_samples + reg * W
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
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_of_samples = X.shape[0]
  scores = X.dot(W)
  scores -= np.max(scores, axis=1).reshape(-1,1)
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1).reshape(-1,1)
  loss = -np.sum(np.log(probs[np.arange(num_of_samples), y]))
  loss = loss/num_of_samples + 0.5 * reg * np.sum(W * W)
  one_hot_y = np.zeros(scores.shape)
  one_hot_y[np.arange(num_of_samples), y] = 1
  probs_mask = probs - one_hot_y
  dW = X.T.dot(probs_mask)
  dW = dW/num_of_samples + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

