import haiku as hk
import jax
import jax.numpy as jnp


class Head(hk.Module):
    def __init__(self, head_size, n_embed=32, block_size=8):
        super(Head, self).__init__()
        self.head_size = head_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.tril = jnp.tril(jnp.ones((block_size, block_size)))

    def __call__(self, x):
        B, T, C = x.shape
        k = hk.Linear(output_size=self.head_size, with_bias=False)(x)
        q = hk.Linear(output_size=self.head_size, with_bias=False)(x)

        weights = q @ jnp.transpose(k, axes=(0, 2, 1)) * C**-0.5
        weights = jnp.where(self.tril[:T, :T] == 0, -jnp.inf, weights)
        weights = jax.nn.softmax(weights, axis=-1)
        v = hk.Linear(output_size=self.head_size, with_bias=False)(x)
        return weights @ v
