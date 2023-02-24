import haiku as hk
import jax

from head import Head, MultiHead


def simple_feed_forward(x, n_embed):
    mlp = hk.Sequential([hk.Linear(output_size=n_embed), jax.nn.relu])
    return mlp(x)


class SimpleBigram(hk.Module):
    def __init__(self, vocab_size, n_embed=32, block_size=8, n_heads=4):
        super(SimpleBigram, self).__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.n_heads = n_heads

    def __call__(self, x):
        B, T = x.shape

        x = hk.Embed(self.vocab_size, self.n_embed)(x)
        pos_emb = hk.Embed(self.block_size, self.n_embed)(jax.numpy.arange(T))
        x = x + pos_emb

        # x = Head(head_size=self.n_embed)(x)

        x = MultiHead(n_heads=self.n_heads, head_size=self.n_embed // self.n_heads)(x)
        x = simple_feed_forward(x, self.n_embed)
        x = hk.Linear(output_size=self.vocab_size)(x)

        return x


def get_model(vocab_size):
    model = hk.without_apply_rng(
        hk.transform(lambda x: SimpleBigram(vocab_size=vocab_size)(x))
    )
    return model
