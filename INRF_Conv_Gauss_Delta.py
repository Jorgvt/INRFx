

from typing import Tuple, Any, Callable

import jax
from jax import lax, random, numpy as jnp
import flax
import flax.linen as nn

from fxlayers.layers import GaussianLayerGamma, pad_same_from_kernel_size

class INRF(nn.Module):
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
        M = GaussianLayerGamma(features=c, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, fs=self.fs, normalize_prob=False, normalize_energy=True, name="m")
        W = GaussianLayerGamma(features=c, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, fs=self.fs, normalize_prob=False, normalize_energy=True, name="w")

        ## Calculate first term
        first_term = M(inputs, **kwargs)

        ## Calculate the differences between all the pixels
        if h % 2 == 0: h_ = h - 1
        else: h_ = h
        if w % 2 == 0: w_ = w - 1
        else: w_ = w
        diff_kernel = self.generate_diff_kernel((b,h_,w_,c))
        difference = lax.conv_general_dilated(
            jnp.transpose(pad_same_from_kernel_size(inputs, (h,w), mode="wrap"), [0, 3, 1, 2]),  # lhs = NCHW image tensor
            jnp.transpose(diff_kernel, [3, 2, 0, 1]),  # rhs = OIHW conv kernel tensor
            (self.strides, self.strides),
            "VALID")
        ## Move the channels back to the last dim
        difference = jnp.transpose(difference, (0, 2, 3, 1))

        ## Assuming that w is the same for every pixel we can take it out of the sum
        second_term = W(difference, **kwargs)

        outputs = first_term - self.l*second_term

        return outputs

    @staticmethod
    def generate_diff_kernel(input_shape):
        b, h, w, c = input_shape
        diff_kernel = -1*jnp.ones(shape=(h,w,c,c))
        for i in range(c):
            diff_kernel = diff_kernel.at[h//2,w//2,i,i].set(1)
        return diff_kernel


if __name__ == "__main__":
    model = INRF(kernel_size=31, fs=32)
    pred, variables = model.init_with_output(random.PRNGKey(42), jnp.ones((4,384,512,1)))
    print(pred.shape)
    print(jax.tree_util.tree_map(lambda x: x.shape, variables))

