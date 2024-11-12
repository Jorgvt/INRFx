
from typing import Tuple, Any

import jax
from jax import lax, random, numpy as jnp
import flax
import flax.linen as nn

class INRF(nn.Module):
    features: int
    kernel_size: Tuple[int, int]
    l: float = 1.
    padding: str = "SAME"
    strides: int = 1

    @nn.compact
    def __call__(self, inputs):
        # inputs -> (H, W, C)
        b, h, w, c = inputs.shape
        outputs = jnp.empty_like(inputs)
        M = self.param("m", nn.initializers.lecun_normal(), (h, w, 1, h, w, 1))
        W = self.param("w", nn.initializers.lecun_normal(), (h, w, 1, h, w, 1))
        G = self.param("g", nn.initializers.lecun_normal(), (h, w, *self.kernel_size, c, self.features))

        outputs = jax.vmap(
            jax.vmap(
                self.per_pixel, in_axes=(0,0,0,None,None,None, None), out_axes=1
            ), in_axes=(1,1,1,None,None,None,None), out_axes=2
        )(M, W, G, inputs, self.l, self.strides, self.padding)

        return outputs

    @staticmethod
    def per_pixel(M, W, G, inputs, l, strides, padding):
        first_term = jnp.sum(M*inputs, axis=(1,2))
        shifted = lax.conv_general_dilated(
            jnp.transpose(inputs, [0, 3, 1, 2]),  # lhs = NCHW image tensor
            jnp.transpose(G, [3, 2, 0, 1]),  # rhs = OIHW conv kernel tensor
            (strides, strides),
            padding)
        ## Move the channels back to the last dim
        shifted = jnp.transpose(shifted, (0, 2, 3, 1))

        second_term = l * jnp.sum(W * nn.relu(inputs - shifted), axis=(1,2))
        return first_term - second_term
