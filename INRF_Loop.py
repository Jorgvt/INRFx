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

        for i in range(h):
            for j in range(w):
                first_term = jnp.sum(M[i,j]*inputs, axis=(1,2))
                shifted = lax.conv_general_dilated(
                    jnp.transpose(inputs, [0, 3, 1, 2]),  # lhs = NCHW image tensor
                    jnp.transpose(G[i,j], [3, 2, 0, 1]),  # rhs = OIHW conv kernel tensor
                    (self.strides, self.strides),
                    self.padding)
                ## Move the channels back to the last dim
                shifted = jnp.transpose(shifted, (0, 2, 3, 1))

                second_term = self.l * jnp.sum(W[i,j] * nn.relu(inputs - shifted), axis=(1,2))
                outputs = outputs.at[:,i,j].set(first_term - second_term)

        return outputs
