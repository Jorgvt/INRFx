
from typing import Tuple, Any, Callable

import jax
from jax import lax, random, numpy as jnp
import flax
import flax.linen as nn

from fxlayers.layers import GaussianLayerGamma

class INRF(nn.Module):
    features: int
    kernel_size: int # Tuple[int, int]
    fs: int
    S: Callable = nn.tanh
    l: float = 1.
    padding: str = "SAME"
    strides: int = 1

    @nn.compact
    def __call__(self, inputs, **kwargs):
        # inputs -> (B, H, W, C)
        b, h, w, c = inputs.shape
        M = GaussianLayerGamma(features=self.features, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, fs=self.fs, normalize_prob=False, normalize_energy=True, name="m")
        W = GaussianLayerGamma(features=self.features, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, fs=self.fs, normalize_prob=False, normalize_energy=True, name="w")
        G = GaussianLayerGamma(features=self.features, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, fs=self.fs, normalize_prob=False, normalize_energy=True, name="g")

        ## Calculate first term
        first_term = M(inputs, **kwargs)

        ## Calculate I(y) - g*I(x)
        blurred_input = G(inputs, **kwargs)

        ## Iterate over positions
        difference = jax.vmap(
                        jax.vmap(
                            jax.vmap(
                                # y is the full batch of images, so we give x null dimensions for broadcasting
                                lambda x,y: self.S(x[:,None,None,None]-y).mean(axis=(1,2,3)), 
                                in_axes=(1,None), out_axes=1
                            ), in_axes=(2,None), out_axes=2
                        ), in_axes=(3,None), out_axes=3
        )(inputs, blurred_input)

        ## Assuming that w is the same for every pixel we can take it out of the sum
        second_term = W(difference, **kwargs)

        outputs = first_term - self.l*second_term

        return outputs

if __name__ == "__main__":
    model = INRF(features=64, kernel_size=31, fs=32)
    pred, variables = model.init_with_output(random.PRNGKey(42), jnp.ones((4,28,28,1)))
    print(pred.shape)
    print(jax.tree_util.tree_map(lambda x: x.shape, variables))

