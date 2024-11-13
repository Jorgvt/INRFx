from INRF import INRF
import numpy as np
import jax
from jax import random, numpy as jnp
from flax import struct
from flax.core import FrozenDict, pop
import flax.linen as nn
from flax.training import train_state
from clu import metrics
import optax
import tensorflow as tf
tf.config.set_visible_devices([], device_type="GPU")
from keras.datasets.mnist import load_data

## Load data
(X_train, Y_train), (X_test, Y_test) = load_data()
X_train = X_train[...,None]/255.
X_test = X_test[...,None]/255.
Y_train = Y_train.astype(np.int32)
Y_test = Y_test.astype(np.int32)

dst_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
dst_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

dst_train_rdy = dst_train.shuffle(buffer_size=50, seed=42, reshuffle_each_iteration=True)\
                         .batch(32, num_parallel_calls=tf.data.AUTOTUNE)\
                         .prefetch(1)
dst_test_rdy = dst_test.batch(32, num_parallel_calls=tf.data.AUTOTUNE)\
                        .prefetch(1)
## Boilerplate
@struct.dataclass
class Metrics(metrics.Collection):
    """Collection of metrics to be tracked during training."""

    loss: metrics.Average.from_output("loss")
    accuracy: metrics.Accuracy


class TrainState(train_state.TrainState):
    metrics: Metrics
    state: FrozenDict


def create_train_state(module, key, tx, input_shape):
    """Creates the initial `TrainState`."""
    variables = module.init(key, jnp.ones(input_shape))
    state, params = pop(variables, "params")
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        state=state,
        tx=tx,
        metrics=Metrics.empty(),
    )

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
        return optax.softmax_cross_entropy_with_integer_labels(pred, Y).mean(), pred

    (loss, pred), grads =  jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics_updates = state.metrics.single_from_model_output(logits=pred, labels=Y, loss=loss)
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics)
    return loss, state

@jax.jit
def compute_metrics(state, batch):
    """Obtaining the metrics for a given batch."""
    X, Y = batch
    def loss_fn(params):
        pred = state.apply_fn({"params":params}, X)
        return optax.softmax_cross_entropy_with_integer_labels(pred, Y).mean(), pred

    loss, pred = loss_fn(state.params)
    metrics_updates = state.metrics.single_from_model_output(logits=pred, labels=Y, loss=loss)
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics)
    return state

## Initialize metrics
metrics_history = {'train_loss':[], 'val_loss':[], 'train_accuracy':[], 'val_accuracy':[]}

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
    print(f"Epoch {epoch} -> [Train] Loss: {metrics_history['train_loss'][-1]} Acc: {metrics_history['train_accuracy'][-1]} | [Val] Loss: {metrics_history['val_loss'][-1]} Acc: {metrics_history['val_accuracy'][-1]} ")
