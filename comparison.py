import jax
from jax import random, numpy as jnp
from INRF_Loop import INRF
from INRF import INRF as INRF_vmap


model = INRF(features=1, kernel_size=(3,3))
inputs = jnp.ones((1,28,28,1))
variables = model.init(random.PRNGKey(42), inputs)
pred = model.apply(variables, inputs)


model = INRF_vmap(features=1, kernel_size=(3,3))
inputs = jnp.ones((1,28,28,1))
variables = model.init(random.PRNGKey(42), inputs)
pred_vmap = model.apply(variables, inputs)

print(jnp.allclose(pred, pred_vmap))
