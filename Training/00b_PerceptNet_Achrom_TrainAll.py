# %%
import os
from typing import Any
import numpy as np
from einops import rearrange
import jax
from jax import random, numpy as jnp
import flax
from flax import struct
from flax.core import freeze, unfreeze, FrozenDict, pop
import flax.linen as nn
from flax.training import train_state
from flax.training import orbax_utils
from clu import metrics
import dm_pix as pix
import optax
import orbax.checkpoint
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.set_visible_devices([], device_type="GPU")

import wandb

from iqadatasets.datasets import *
from paramperceptnet.layers import *
from paramperceptnet.training import train_step, compute_metrics, pearson_correlation
from paramperceptnet.configs import param_config as config
from initialization import humanlike_init
from layers import *
from fxlayers.layers import pad_same_from_kernel_size

## Config
config["BATCH_SIZE"] = 8
config["EPOCHS"] = 200

wandb.init(project="PerceptNet_INRF",
           job_type="training",
           mode="disabled",
           name="PerceptAchrom_TrainAll",
           config=config)
# config = wandb.run.config
print(config)

# %%
## Load data
dst_train = TID2008("/media/disk/vista/BBDD_video_image/Image_Quality/TID/TID2008/", exclude_imgs=[25]).dataset
dst_test = TID2013("/media/disk/vista/BBDD_video_image/Image_Quality/TID/TID2013/", exclude_imgs=[25]).dataset

# %%
def preprocess(img, dist, mos):
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, size=(384//1, 512//1))
    dist = tf.image.rgb_to_grayscale(dist)
    dist = tf.image.resize(dist, size=(384//1, 512//1))
    return img, dist, mos

# %%
dst_train_rdy = dst_train.shuffle(buffer_size=50, seed=42, reshuffle_each_iteration=True)\
                         .batch(config.BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE)\
                         .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)\
                         .prefetch(1)
dst_test_rdy = dst_test.batch(config.BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE)\
                        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)\
                        .prefetch(1)

# %%
img, dist, mos = next(iter(dst_train_rdy))
print(img.shape, dist.shape, mos.shape)

# %%
## Boilerplate
@struct.dataclass
class Metrics(metrics.Collection):
    """Collection of metrics to be tracked during training."""
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics
    state: FrozenDict


def create_train_state(module, key, tx, input_shape):
    """Creates the initial `TrainState`."""
    variables = module.init(key, jnp.ones(input_shape))
    state, params = pop(variables, "params")
    return TrainState.create(
        apply_fn=module.apply,
        params=freeze(params),
        state=freeze(state),
        tx=tx,
        metrics=Metrics.empty(),
    )

# %%
class PerceptNet(nn.Module):
    """IQA model inspired by the visual system."""

    config: Any

    @nn.compact
    def __call__(
        self,
        inputs,  # Assuming fs = 128 (cpd)
        **kwargs,
    ):
        ## (Independent) Color equilibration (Gamma correction)
        ## Might need to be the same for each number
        ## bias = 0.1 / kernel = 0.5
        if self.config.USE_GAMMA:
            outputs = GDNGamma()(inputs)
        else:
            outputs = GDN(kernel_size=(1, 1), apply_independently=True)(inputs)

        ## Color (ATD) Transformation
        # outputs = nn.Conv(features=3, kernel_size=(1, 1), use_bias=False, name="Color")(
        #     outputs
        # )
        outputs = nn.max_pool(outputs, window_shape=(2, 2), strides=(2, 2))

        ## GDN Star A - T - D [Separated]
        outputs = GDN(kernel_size=(1, 1), apply_independently=True)(outputs)

        ## Center Surround (DoG)
        outputs = pad_same_from_kernel_size(
            outputs, kernel_size=self.config.CS_KERNEL_SIZE, mode="symmetric"
        )
        outputs = CenterSurroundLogSigmaK(
            features=1,
            kernel_size=self.config.CS_KERNEL_SIZE,
            fs=21,
            use_bias=False,
            padding="VALID",
        )(outputs, **kwargs)
        outputs = nn.max_pool(outputs, window_shape=(2, 2), strides=(2, 2))

        ## GDN per channel with mean substraction in T and D (Spatial Gaussian Kernel)
        ### fs = 32 / kernel_size = (11,11) -> 0.32 > 0.02 --> OK!
        ## TO-DO: - Spatial Gaussian Kernel (0.02 deg) -> fs = 64/2 & 0.02*64/2 = sigma (px) = 0.69
        outputs = GDNGaussian(
            kernel_size=self.config.GDNGAUSSIAN_KERNEL_SIZE,
            apply_independently=True,
            fs=32,
            padding="symmetric",
            normalize_prob=self.config.NORMALIZE_PROB,
            normalize_energy=self.config.NORMALIZE_ENERGY,
        )(outputs, **kwargs)

        ## GaborLayer per channel with GDN mixing only same-origin-channel information
        ### [Gaussian] sigma = 0.2 (deg) fs = 32 / kernel_size = (21,21) -> 21/32 = 0.66 --> OK!
        outputs = pad_same_from_kernel_size(
            outputs, kernel_size=self.config.GABOR_KERNEL_SIZE, mode="symmetric"
        )
        # outputs, fmean, theta_mean = GaborLayerGamma_(n_scales=4+2+2, n_orientations=8*3, kernel_size=self.config.GABOR_KERNEL_SIZE, fs=32, xmean=self.config.GABOR_KERNEL_SIZE/32/2, ymean=self.config.GABOR_KERNEL_SIZE/32/2, strides=1, padding="VALID", normalize_prob=self.config.NORMALIZE_PROB, normalize_energy=self.config.NORMALIZE_ENERGY, zero_mean=self.config.ZERO_MEAN, use_bias=self.config.USE_BIAS, train_A=self.config.A_GABOR)(outputs, return_freq=True, return_theta=True, **kwargs)
        outputs, fmean, theta_mean = GaborLayerGammaHumanLike_(
            n_scales=[4],
            n_orientations=[8],
            kernel_size=self.config.GABOR_KERNEL_SIZE,
            fs=32,
            xmean=self.config.GABOR_KERNEL_SIZE / 32 / 2,
            ymean=self.config.GABOR_KERNEL_SIZE / 32 / 2,
            strides=1,
            padding="VALID",
            normalize_prob=self.config.NORMALIZE_PROB,
            normalize_energy=self.config.NORMALIZE_ENERGY,
            zero_mean=self.config.ZERO_MEAN,
            use_bias=self.config.USE_BIAS,
            train_A=self.config.A_GABOR,
        )(outputs, return_freq=True, return_theta=True, **kwargs)

        ## Final GDN mixing Gabor information (?)
        outputs = GDNSpatioChromaFreqOrient(
            # kernel_size=self.config.GDNFINAL_KERNEL_SIZE,
            kernel_size=21,
            strides=1,
            padding="symmetric",
            fs=32,
            apply_independently=False,
            normalize_prob=self.config.NORMALIZE_PROB,
            normalize_energy=self.config.NORMALIZE_ENERGY,
        )(outputs, fmean=fmean, theta_mean=theta_mean, **kwargs)

        return outputs
# %%
def check_trainable_fn(tree, use_bool=False):
    def check_trainable(path):
        return "trainable" if not use_bool else True
    return freeze(flax.traverse_util.path_aware_map(lambda path,v: check_trainable(path), tree))

# %%
state = create_train_state(PerceptNet(config=config), random.PRNGKey(42),
                           tx=optax.multi_transform({"trainable": optax.adam(3e-4), "non-trainable": optax.set_to_zero()}, check_trainable_fn),
                           input_shape=(1,*img.shape[1:]))
pred, s_ = state.apply_fn({"params": state.params, **state.state}, jnp.ones((1,*img.shape[1:])), train=True, mutable=list(state.state.keys()))
state = state.replace(state=s_)
total_parameters = jax.tree_util.tree_map(lambda x: x.size, state.params)
total_parameters = sum(jax.tree_util.tree_leaves(total_parameters))
trainable_parameters = jax.tree_util.tree_map(lambda x,flag: x.size if flag else 0, state.params, check_trainable_fn(state.params, use_bool=True))
trainable_parameters = sum(jax.tree_util.tree_leaves(trainable_parameters))
print(f"Trainable parameters: {trainable_parameters} | Total parameters: {total_parameters}")
print(check_trainable_fn(state.params))

# %%
state.params

# %%
# %%
params = unfreeze(state.params)
## Initialize PerceptNet params
params = humanlike_init(params)
state = state.replace(params=freeze(params))
pred, s_ = state.apply_fn({"params": state.params, **state.state}, img.numpy()[:1], train=True, mutable=list(state.state.keys()))
state = state.replace(state=s_)


## Save model at init time
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(state)
orbax_checkpointer.save(
    os.path.join(wandb.run.dir, "model-0"), state, save_args=save_args, force=True
)

## Obtain metrics at init time
### Train
for batch in dst_train_rdy.as_numpy_iterator():
    state = compute_metrics(state=state, batch=batch)
### Log the metrics
for name, value in state.metrics.compute().items():
    train_init = value
### Empty the metrics
state = state.replace(metrics=state.metrics.empty())

### Validation
for batch in dst_test_rdy.as_numpy_iterator():
    state = compute_metrics(state=state, batch=batch)
### Empty the metrics
for name, value in state.metrics.compute().items():
    val_init = value
### Empty the metrics
state = state.replace(metrics=state.metrics.empty())

wandb.log(
        {
            "epoch": 0,
            "learning_rate": config.LEARNING_RATE,
            "train_loss": train_init,
            "val_loss": val_init,
        }
    )

# %%
def forward(state, img):
    pred = state.apply_fn({"params": state.params, **state.state}, img)
    return pred

# %%
pred = forward(state, img[:1].numpy())
print(f"Pred: {pred.shape}")

# %%
## Initialize metrics
metrics_history = {'train_loss':[], 'val_loss':[]}

# %%
## Training Loop
for epoch in range(config.EPOCHS):
    ## Training ##
    for batch in dst_train_rdy.as_numpy_iterator():
        state = train_step(state, batch)
        # break

    ## Log the metrics
    for name, value in state.metrics.compute().items():
        metrics_history[f"train_{name}"].append(value)

    ## Empty the metrics
    state = state.replace(metrics=state.metrics.empty())

    ## Validation ##
    for batch in dst_test_rdy.as_numpy_iterator():
        state = compute_metrics(state=state, batch=batch)
        # break

    ## Log the metrics
    for name, value in state.metrics.compute().items():
        metrics_history[f"val_{name}"].append(value)

    ## Empty the metrics
    state = state.replace(metrics=state.metrics.empty())

    ## Checkpointing
    if metrics_history["val_loss"][-1] <= min(metrics_history["val_loss"]):
        orbax_checkpointer.save(
            os.path.join(wandb.run.dir, "model-best"),
            state,
            save_args=save_args,
            force=True,
        )

    ## Logging ##
    wandb.log(
            {
                "epoch": epoch + 1,
                "learning_rate": config.LEARNING_RATE,
                **{name: values[-1] for name, values in metrics_history.items()},
            }
        )
    print(f"Epoch {epoch} -> [Train] Loss: {metrics_history['train_loss'][-1]} | [Val] Loss: {metrics_history['val_loss'][-1]}")

## Save final model
orbax_checkpointer.save(
    os.path.join(wandb.run.dir, "model-final"), state, save_args=save_args
)
wandb.finish()
