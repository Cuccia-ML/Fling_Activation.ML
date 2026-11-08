import numpy as np
import time
from scipy.special import lambertw

def create_architecture(net_arch):
  #Create the Parameters

  weights = []
  biases = []
  for i in range(len(net_arch) - 1):
    w = np.random.uniform(low=-(1 / net_arch[i]), high=(1 / net_arch[i]), size=(net_arch[i], net_arch[i + 1]))
    weights.append(w)

    b = np.random.uniform(low=0, high=0, size=(1, net_arch[i + 1]))
    biases.append(b)

  return weights, biases

def get_t(a, C):
#Get the turning point (t) from the hyperparameters a and C
    lam_w = lambertw(-a * C, k=0)
    t_real = np.real(lam_w)
    t = -float(t_real) / a

    return t

def fling_forward(x, a, C, t):
    #Activation function fowardpass for Fling, max(Ce^(ax), x)
    exp_branch = C * np.exp(a * x)
    f_x = np.maximum(exp_branch, x)

    mask = (f_x == x)

    return f_x, mask

def fling_backward(grad_output, mask, f_x, a):
    """Activation Function backwardprop for Fling, stored with
      'mask', 1 = linear branch, linear derivative is 1. 0 = exponential branch,
      exponential branch = a * f(x)"""

    derive_linear = 1.0
    derive_exp = a * f_x

    activation = np.where(mask, grad_output * derive_linear, grad_output * derive_exp)

    return activation

def forwardpass(hyperparameters, weights, biases, inputs):

  a = hyperparameters[0]
  C = hyperparameters[1]
  t = hyperparameters[2]

  hidden = [inputs]
  hidden_mask = [None]
  for i in range(len(biases)):
    w = weights[i]
    b = biases[i]

    hidden_i = np.dot(hidden[i], w) + b

    hidden_i, mask = fling_forward(hidden_i, a, C, t)

    hidden.append(hidden_i)
    hidden_mask.append(mask)

  return hidden, hidden_mask

def backwardprop(hyperparameters, weights, biases, inputs, targets, mask, outputs):
  a = hyperparameters[0]
  C = hyperparameters[1]
  t = hyperparameters[2]

  out_grad = outputs[len(outputs) - 1] - targets
  out_grad = fling_backward(out_grad, mask[len(mask) - 1], outputs[len(outputs) - 1], a)

  h_grad = [None for _ in range(len(biases))]
  h_grad[-1] = out_grad

  w_grad = [None for _ in range(len(weights))]

  for i in range(len(biases) - 1, -1, -1):
    w = weights[i]
    b = biases[i]

    if i > 0:
      h_grad_prev = np.dot(h_grad[i], w.T)
      h_grad_prev = fling_backward(h_grad_prev, mask[i], outputs[i], a)
      h_grad[i - 1] = h_grad_prev

    w_grad_i = np.dot(outputs[i].T, h_grad[i])
    w_grad[i] = w_grad_i

  return h_grad, w_grad

def update_params(weights, biases, h_grad, w_grad, learning_rate):
  for i in range(len(biases)):
    weights[i] -= learning_rate * w_grad[i]
    biases[i] -= learning_rate * h_grad[i]

  return weights, biases

#Inputs & Targets (Defined explicitly as 2D row vectors)
inputs_data = [[0.2, 0.5], [-0.3, 0.3]]
target_data = [[0.1, 0.2], [0.9, 1]]

#Training Control dials
EPOCHS = 1000
learning_rate = 0.1

net_arch = [2, 50, 80, 2]

#Hyperparameters
a = 2
C = -1
t = get_t(a, C)

hyperparameters = [a, C, t]

#Create the network
weights, biases = create_architecture(net_arch)

for epoch in range(EPOCHS):
  for data in range(len(inputs_data)):

    inputs = np.array([inputs_data[data]])
    targets = np.array([target_data[data]])

    outputs, mask = forwardpass(hyperparameters, weights, biases, inputs)
    h_gradients, w_gradients = backwardprop(hyperparameters, weights, biases, inputs, targets, mask, outputs)
    weights, biases = update_params(weights, biases, h_gradients, w_gradients, learning_rate)

for data in range(len(inputs_data)):

    inputs = np.array([inputs_data[data]])
    targets = np.array([target_data[data]])

    outputs, mask = forwardpass(hyperparameters, weights, biases, inputs)
    last = outputs[len(outputs) - 1]

    print(f"\nInputs: {inputs}, Predictions: {last}, Error: {last - targets}")
