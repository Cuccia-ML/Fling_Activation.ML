import matplotlib.pyplot as plt

C = -7
a = 15

def get_t(a, C):
    lam_w = lambertw(-a * C, k=0)
    t_real = np.real(lam_w)
    t = -float(t_real) / a

    return t

t = get_t(a, C)
# Calculate final pre-activations
z_final = np.dot(X, W)

# Generate values to plot the activation function
z_range = np.linspace(-3, 3, 500)
f_range = np.maximum(C * np.exp(a * z_range), z_range)

# Plot the activation function and the distribution of our actual pre-activations
plt.figure(figsize=(10, 5))
plt.plot(z_range, f_range, label='Activation f(z)', color='blue', linewidth=2)
plt.hist(z_final, bins=20, density=True, alpha=0.6, color='orange', label='Current z distribution (XW)')
plt.axvline(t, color='red', linestyle='--', label=f'Turning point t ({t:.4f})')
plt.title('Activation Function Behavior vs. Current Pre-activation Distribution')
plt.xlabel('z')
plt.ylabel('f(z) / Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
