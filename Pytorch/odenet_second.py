import jax.numpy as jnp
import numpy.random as npr
from jax import jit, grad
import matplotlib.pyplot as plt
from jax.experimental.ode import odeint
from jax import vmap


epsilon_Euler = 0.01

def mlp(params, inputs_mlp):
    # A multi-layer perceptron, i.e. a fully-connected neural network.
    for w, b in params:
        outputs_mlp = jnp.dot(inputs_mlp, w) + b  # Linear transform
        inputs_mlp = jnp.tanh(outputs_mlp)  # Nonlinearity
    return epsilon_Euler*outputs_mlp


def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
    return [(scale * rng.randn(m, n), scale * rng.randn(n))
            for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]


def nn_dynamics(state, time, params):
    state_and_time = jnp.hstack([state, jnp.array(time)])
    return mlp(params, state_and_time)


def odenet(params, input):
    start_and_end_times = jnp.array([0.0, 1.0])
    init_state, final_state = odeint(nn_dynamics, input, start_and_end_times, params)
    return final_state


def to_np(x):
    return x.detach().cpu().numpy()


batched_odenet = vmap(odenet, in_axes=(None, 0))

dim = 2
if dim == 1:
    # Toy 1D dataset.
    inputs = jnp.reshape(jnp.linspace(-2.0, 2.0, 20), (20, 1))
    targets = inputs ** 3 + 0.1 * inputs
elif dim == 2:
    # Toy 2D dataset
    x = jnp.linspace(0, 1, 5)
    y = jnp.linspace(0, 1, 5)
    xv, yv = jnp.meshgrid(x, y)
    inputs = jnp.transpose(jnp.array([xv.flatten(), yv.flatten()]))
    targets = jnp.transpose(jnp.array([inputs[:, 0]**3, inputs[:, 0]*2-inputs[:, 1]]))

# Hyperparameters.
param_scale = 1.0
step_size = 0.01
train_iters = 100000

# We need to change the input dimension to 2, to allow time-dependent dynamics.
if dim == 1:
    odenet_layer_sizes = [2, 10, 1]
elif dim == 2:
    odenet_layer_sizes = [3, 10, 10, 2]


def odenet_loss(params, inputs, targets):
    preds = batched_odenet(params, inputs)
    return jnp.mean(jnp.sum(jnp.sum((preds - targets) ** 2, axis=1), axis=0))


@jit
def odenet_update(params, inputs, targets):
    grads = grad(odenet_loss)(params, inputs, targets)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]


# Initialize and train ODE-Net.
odenet_params = init_random_params(param_scale, odenet_layer_sizes)

random_output = batched_odenet(odenet_params, inputs)

for i in range(train_iters):
    if dim == 1:
        inputs_train = jnp.reshape(npr.choice(jnp.squeeze(inputs), size=10), (10, 1))
        targets_train = inputs_train ** 3 + 0.1 * inputs_train
        odenet_params = odenet_update(odenet_params, inputs_train, targets_train)
    if dim == 2:
        odenet_params = odenet_update(odenet_params, inputs, targets)

trained_output = batched_odenet(odenet_params, inputs)
# Plot resulting model.
if dim == 1:
    fig = plt.figure(figsize=(6, 4), dpi=150)
    ax = fig.gca()
    ax.scatter(inputs, targets, lw=0.5, color='green')
    ax.scatter(inputs_train, targets_train, lw=0.5, color='blue')
    fine_inputs = jnp.reshape(jnp.linspace(-3.0, 3.0, 100), (100, 1))
    # ax.plot(fine_inputs, resnet(resnet_params, fine_inputs, resnet_depth), lw=0.5, color='blue')
    ax.plot(fine_inputs, batched_odenet(odenet_params, fine_inputs), lw=0.5, color='red')
    ax.set_xlabel('input')
    ax.set_ylabel('output')
    plt.legend(('Data', 'Training data', 'Resnet predictions', 'ODE Net predictions'))
if dim == 2:
    size_fine = 200
    x = jnp.linspace(0, 1, size_fine)
    y = jnp.linspace(0, 1, size_fine)
    xv, yv = jnp.meshgrid(x, y)
    fine_inputs = jnp.transpose(jnp.array([xv.flatten(), yv.flatten()]))
    fine_outputs = batched_odenet(odenet_params, fine_inputs)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(inputs[:, 0], inputs[:, 1], targets[:, 0], lw=0.5, color='green')
    ax.scatter3D(fine_inputs[:, 0], fine_inputs[:, 1], fine_outputs[:, 0], c=fine_outputs[:, 0], cmap='Reds')
    plt.legend(('Data', 'ODE Net predictions'))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(inputs[:, 0], inputs[:, 1], targets[:, 1], lw=0.5, color='green')
    ax.scatter3D(fine_inputs[:, 0], fine_inputs[:, 1], fine_outputs[:, 1], c=fine_outputs[:, 0], cmap='Reds')
    plt.legend(('Data', 'ODE Net predictions'))
print('End')
