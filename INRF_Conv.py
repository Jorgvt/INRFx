from typing import Tuple, Any, Callable

import jax
from jax import lax, random, numpy as jnp
import flax
import flax.linen as nn

class INRF(nn.Module):
    features: int
    kernel_size: Tuple[int, int]
    S: Callable = nn.tanh
    l: float = 1.
    padding: str = "SAME"
    strides: int = 1

    @nn.compact
    def __call__(self, inputs):
        # inputs -> (H, W, C)
        b, h, w, c = inputs.shape
        M = self.param("m", nn.initializers.lecun_normal(), (*self.kernel_size, c, self.features))
        W = self.param("w", nn.initializers.lecun_normal(), (*self.kernel_size, c, self.features))
        G = self.param("g", nn.initializers.lecun_normal(), (*self.kernel_size, c, self.features))

        ## Calculate first term
        first_term = lax.conv_general_dilated(
            jnp.transpose(inputs, [0, 3, 1, 2]),  # lhs = NCHW image tensor
            jnp.transpose(M, [3, 2, 0, 1]),  # rhs = OIHW conv kernel tensor
            (self.strides, self.strides),
            self.padding)
        ## Move the channels back to the last dim
        first_term = jnp.transpose(first_term, (0, 2, 3, 1))

        ## Calculate I(y) - g*I(x)
        blurred_input = lax.conv_general_dilated(
            jnp.transpose(inputs, [0, 3, 1, 2]),  # lhs = NCHW image tensor
            jnp.transpose(G, [3, 2, 0, 1]),  # rhs = OIHW conv kernel tensor
            (self.strides, self.strides),
            self.padding)
        ## Move the channels back to the last dim
        blurred_input = jnp.transpose(blurred_input, (0, 2, 3, 1))

        ## Iterate over positions
        difference = jax.vmap(
                        jax.vmap(
                            jax.vmap(
                                # y is the full batch of images, so we give x null dimensions for broadcasting
                                lambda x,y: self.S(x[:,None,None,None]-y).sum(axis=(1,2,3)), 
                                in_axes=(1,None), out_axes=1
                            ), in_axes=(2,None), out_axes=2
                        ), in_axes=(3,None), out_axes=3
        )(inputs, blurred_input)

        ## Assuming that w is the same for every pixel we can take it out of the sum
        second_term = lax.conv_general_dilated(
            jnp.transpose(difference, [0, 3, 1, 2]),  # lhs = NCHW image tensor
            jnp.transpose(W, [3, 2, 0, 1]),  # rhs = OIHW conv kernel tensor
            (self.strides, self.strides),
            self.padding)
        ## Move the channels back to the last dim
        second_term = jnp.transpose(second_term, (0, 2, 3, 1))

        outputs = first_term - self.l*second_term

        return outputs

if __name__ == "__main__":
    model = INRF(features=64, kernel_size=(3,3))
    pred, variables = model.init_with_output(random.PRNGKey(42), jnp.ones((4,28,28,1)))
    print(pred.shape)
    print(jax.tree_util.tree_map(lambda x: x.shape, variables))

