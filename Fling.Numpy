import numpy as np
import time
from scipy.special import lambertw

#Training Control dials
EPOCHS = 1000
learning_rate = 0.1

#Hyperparameters
a = 2
C = -1

#Seed
np.random.seed(82)

X = np.random.randn(64, 10)
y_true = np.random.randn(64, 1)

W = np.random.randn(10, 1) * 0.1

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

def Training_Loop(t, W):
    #Network training loop
    print("\n--- Starting Loop ---")

    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS):

        z = np.dot(X, W)

        f_z, mask = fling_forward(z, a, C, t)

        loss = np.mean((f_z - y_true) ** 2)

        grad_loss_out = 2.0 * (f_z - y_true) / y_true.size
        grad_activation = fling_backward(grad_loss_out, mask, f_z, a)
        grad_W = np.dot(X.T, grad_activation)

        W -= learning_rate * grad_W

        grad_norm = np.linalg.norm(grad_W)
        print(f"Epoch {epoch}/{EPOCHS - 1} : Loss: {loss:.6f}")

    end_time = time.perf_counter()
    print(f"\nTraining completed in {(end_time - start_time)*1000:.2f} ms")

    return W

#Handling the actual loop starting
t = get_t(a, C)
W = Training_Loop(t, W)
