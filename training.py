from jaxlib.xla_extension import buffer_to_dlpack_managed_tensor
from INRF import INRF
import jax
from jax import random, numpy as jnp
import flax.linen as nn
import optax
import tensorflow as tf
from keras.datasets.mnist import load_data

from paramperceptnet.training import create_train_state

## Load data
(X_train, Y_train), (X_test, Y_test) = load_data()
X_train = X_train[...,None]/255.
X_test = X_test[...,None]/255.
# Y_train = Y_train[...,None]
# Y_test = Y_test[...,None]

dst_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
dst_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

dst_train_rdy = dst_train.shuffle(buffer_size=50, seed=42, reshuffle_each_iteration=True)\
                         .batch(32, num_parallel_calls=tf.data.AUTOTUNE)\
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
    return loss, state

## Training Loop
epochs = 10
i = 0
for epoch in range(epochs):
    for batch in dst_train_rdy.as_numpy_iterator():
        loss, state = train_step(state, batch)
        if i % 50 == 0: print(loss)
        i += 1

