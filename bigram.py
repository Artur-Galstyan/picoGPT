import haiku as hk
import jax

from head import Head


class SimpleBigram(hk.Module):
    def __init__(self, vocab_size, n_embed=32, block_size=8):
        super(SimpleBigram, self).__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.block_size = block_size

    def __call__(self, x):
        B, T = x.shape

        x = hk.Embed(self.vocab_size, self.n_embed)(x)
        pos_emb = hk.Embed(self.block_size, self.n_embed)(jax.numpy.arange(T))
        x = x + pos_emb
    
        x = Head(head_size=self.n_embed)(x)

        x = hk.Linear(output_size=self.vocab_size)(x)

        return x


def get_model(vocab_size):
    model = hk.without_apply_rng(
        hk.transform(lambda x: SimpleBigram(vocab_size=vocab_size)(x))
    )
    return model
