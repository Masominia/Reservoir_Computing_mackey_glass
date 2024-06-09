import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from echo_state_network import ESN
from chaotic_time_series import mackey_glass

# Set random seeds for reproducibility
np.random.seed(64)
rng = np.random.RandomState(100)

# Linear regression function
def linreg(S, D):
    return (np.linalg.pinv(S) @ D).T

# Ridge regression function
def Ridge(S, D, alpha=0.1):
    return (np.linalg.pinv(S.T @ S + alpha ** 2 * np.eye(S.shape[1])) @ (S.T @ D)).T

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity(x):
    return x

def tanh(x):
    return np.tanh(x)

# Encoding function
def encode(x):
    return np.exp(1j * np.pi * x)

# Mean square error function
def mean_square_error(predict, expect):
    return np.sqrt(np.mean((predict - expect)**2) / np.var(expect))

# Weight initialization functions
def generate_W_res(num_nodes):
    internal_weights = np.random.choice([0, 0.4, -0.4], (num_nodes, num_nodes), p=[0.95, 0.025, 0.025])
    maxval = max(abs(linalg.eigvals(internal_weights)))
    return internal_weights / maxval * 0.95

def generate_W_in(num_post_nodes, num_pre_nodes):
    return np.random.choice([0, 0.14, -0.14], (num_post_nodes, num_pre_nodes), p=[0.5, 0.25, 0.25])

def generate_W_fb(num_pre_nodes, num_post_nodes):
    return np.random.rand(num_pre_nodes, num_post_nodes) * 1.12 - 0.56

def generate_W_out(num_post_nodes, num_pre_nodes):
    return np.zeros((num_post_nodes, num_pre_nodes))

def generate_variational_weights_optical(num_post_nodes, num_pre_nodes):
    weights_real = np.random.normal(0.0, 1.0, (num_post_nodes, num_pre_nodes))
    weights_imag = np.random.normal(0.0, 1.0, (num_post_nodes, num_pre_nodes))
    return weights_real + 1j * weights_imag

def generate_W_res_optical(num_nodes):
    weights_real = np.random.normal(0.0, 1.0, (num_nodes, num_nodes))
    weights_imag = np.random.normal(0.0, 1.0, (num_nodes, num_nodes))
    weights = weights_real + 1j * weights_imag
    spectral_radius = max(abs(linalg.eigvals(weights)))
    return (weights / spectral_radius) * 0.99

# Constants and parameters
ninternal, ninput, noutput = 400, 1, 1
initlen, train_size, sample_size = 2000, 2000, 6000
test_size = sample_size - train_size

# Initialize weights
noise = np.random.choice([0.00001, -0.00001], p=[0.5, 0.5])
W = generate_W_res(ninternal)
W_in = generate_W_in(ninternal, ninput)
W_fb = generate_W_fb(ninternal, noutput)
W_out = generate_W_out(noutput, ninternal + ninput)

# Mackey Glass data
u_train = np.full((train_size,), 0.5)
MG_y = mackey_glass(sample_len=sample_size, tau=17, n_samples=1).reshape(sample_size,)
MG_y_train, MG_y_test = MG_y[:train_size], MG_y[train_size:]
u_test = np.full((test_size,), 0.5)

# Initialize and train ESN model
model = ESN(ninput=ninput, ninternal=ninternal, noutput=noutput, W=W, W_in=W_in, W_fb=W_fb, W_out=W_out,
            activation=tanh, out_activation=identity, invout_activation=identity, encode=encode,
            spectral_radius=1, dynamics='leaky', regression=Ridge,
            noise_level=noise, delta=1, C=0.44, leakage=0.9)

model_state = model.fit(inputs=u_train, outputs=MG_y_train, nforget=initlen)
MG_y_trained = model.trained_outputs(inputs=u_train, outputs=MG_y_train)
y_predict = model.predict(inputs=u_test, turnoff_noise=True, continuing=True)

# Compute mean square error
MSE = mean_square_error(y_predict, MG_y_test)

# Plot results
plot_size = train_size
plt.plot(np.arange(plot_size), MG_y_train, label="inputs", color="dodgerblue")
plt.plot(np.arange(plot_size), MG_y_trained, linestyle="dashed", label='trained_outputs', color="orange")
plt.plot(np.arange(plot_size, plot_size + test_size), MG_y_test, color="dodgerblue")
plt.plot(np.arange(plot_size, plot_size + test_size), y_predict, linestyle="dashed", label="predict", color="green")
plt.axvline(x=plot_size, label="end of train", color="red")
plt.xlim([train_size - 500, train_size + 500])
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0, fontsize=10)
plt.title('Mackey Glass Prediction')
plt.xlabel("time")
plt.ylabel("o(t)")
plt.show()
