import haiku as hk
import jax


class SimpleBigram(hk.Module):
    def __init__(self, vocab_size):
        super(SimpleBigram, self).__init__()
        self.vocab_size = vocab_size

    def __call__(self, x):
        x = hk.Embed(self.vocab_size, self.vocab_size)(x)
        
        return x


def get_model(vocab_size):
    model = hk.without_apply_rng(
        hk.transform(lambda x: SimpleBigram(vocab_size=vocab_size)(x))
    )
    return model
