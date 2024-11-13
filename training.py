from INRF import INRF
import jax
from jax import random, numpy as jnp
import flax.linen as nn
import optax
import tensorflow as tf
tf.config.set_visible_devices([], device_type="GPU")
from keras.datasets.mnist import load_data

from paramperceptnet.training import create_train_state

## Load data
(X_train, Y_train), (X_test, Y_test) = load_data()
X_train = X_train[...,None]/255.
X_test = X_test[...,None]/255.

dst_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
dst_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

dst_train_rdy = dst_train.shuffle(buffer_size=50, seed=42, reshuffle_each_iteration=True)\
                         .batch(32, num_parallel_calls=tf.data.AUTOTUNE)\
                         .prefetch(1)
dst_test_rdy = dst_test.batch(32, num_parallel_calls=tf.data.AUTOTUNE)\
                        .prefetch(1)

## Model
class Model(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        outputs = INRF(features=64, kernel_size=(3,3))(inputs)
        outputs = nn.Dense(10)(outputs.mean(axis=(1,2)))
        return outputs

state = create_train_state(Model(), random.PRNGKey(42), tx=optax.adam(3e-4), input_shape=X_train[:1].shape)

@jax.jit
def train_step(state, batch):
    X, Y = batch
    def loss_fn(params):
        pred = state.apply_fn({"params":params}, X)
        return optax.softmax_cross_entropy_with_integer_labels(pred, Y).mean()
    loss, grads =  jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics_updates = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics)
    return loss, state

@jax.jit
def compute_metrics(state, batch):
    """Obtaining the metrics for a given batch."""
    X, Y = batch
    def loss_fn(params):
        pred = state.apply_fn({"params":params}, X)
        return optax.softmax_cross_entropy_with_integer_labels(pred, Y).mean()

    metrics_updates = state.metrics.single_from_model_output(loss=loss_fn(state.params))
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics)
    return state

## Initialize metrics
metrics_history = {'train_loss':[], 'val_loss':[]}

## Training Loop
epochs = 10
for epoch in range(epochs):
    ## Training ##
    for batch in dst_train_rdy.as_numpy_iterator():
        loss, state = train_step(state, batch)

    ## Log the metrics
    for name, value in state.metrics.compute().items():
        metrics_history[f"train_{name}"].append(value)

    ## Empty the metrics
    state = state.replace(metrics=state.metrics.empty())

    ## Validation ##
    for batch in dst_test_rdy.as_numpy_iterator():
        state = compute_metrics(state, batch)

    ## Log the metrics
    for name, value in state.metrics.compute().items():
        metrics_history[f"val_{name}"].append(value)

    ## Empty the metrics
    state = state.replace(metrics=state.metrics.empty())

    ## Logging ##
    print(f"Epoch {epoch} -> [Train] Loss: {metrics_history['train_loss'][-1]} | [Val] Loss: {metrics_history['val_loss'][-1]}")
