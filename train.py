try:
    import cPickle as pickle
except:
    import pickle
import os
from tqdm import tqdm
import jax
import jax.numpy as jnp
import optax
import haiku as hk
import numpy as np
from datahandler import get_data
from bigram import get_model

train_dataloader, test_dataloader, vocab_size, encode, decode = get_data()

# print(vocab_size)
bigram_model = get_model(vocab_size)

dummy = next(iter(train_dataloader))

dummy_data = dummy[0].numpy()
dummy_labels = dummy[1].numpy()

# print(f"dummy_data.shape={dummy_data.shape}")
# print(f"dummy_labels.shape={dummy_labels.shape}")

key = hk.PRNGSequence(42)
rng = next(key)
params = bigram_model.init(rng, x=dummy_data)

output = bigram_model.apply(params, x=dummy_data)
# print(output.shape)

cross_entropy_loss = optax.softmax_cross_entropy_with_integer_labels(
    output, dummy_labels
)
# print(cross_entropy_loss.mean())


def generate(idx, max_new_tokens, print_intermediate=False):
    rng_key = hk.PRNGSequence(42)
    for _ in range(max_new_tokens):
        logits = bigram_model.apply(params, x=idx)

        logits = logits[:, -1, :]

        # probs = jax.nn.softmax(logits, axis=-1)

        # print(idx_next)
        key = next(rng_key)
        idx_next = jax.random.categorical(key, logits).reshape((-1, 1))

        # print(idx_next.shape)
        idx = jnp.concatenate((idx, idx_next), axis=1)
        if print_intermediate:
            print(decode(idx[0].tolist()))
        # print(idx.shape)
    return idx


text = generate(jnp.zeros((1, 1), dtype=jnp.int32), 100)[0].tolist()
print(decode(text))


# generate(dummy_data, 1)

# training the bigram

model = lambda params, batch: bigram_model.apply(params, batch)


def loss(params: optax.Params, batch: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    y_hat = model(params, batch)
    loss_value = optax.softmax_cross_entropy_with_integer_labels(y_hat, labels)

    return loss_value.mean()


def fit(params: optax.Params, optim: optax.GradientTransformation) -> optax.Params:
    opt_state = optim.init(params)

    @jax.jit
    def step(params, opt_state, batch, labels):
        loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for i, (x, y) in tqdm(enumerate(train_dataloader)):
        params, opt_state, loss_value = step(params, opt_state, x.numpy(), y.numpy())
        if i % 25000 == 0:
            print(f"step {i}, loss: {loss_value}")

    with open("model/model.pkl", "wb") as f:
        pickle.dump(params, f)

    return params


optimiser = optax.adamw(learning_rate=1e-3)

if os.path.exists("model/model.pkl"):
    with open("model/model.pkl", "rb") as f:
        if f:
            print("Loading model")
            params = pickle.load(f)
else:
    print("training model")
    params = fit(params, optimiser)


text = generate(jnp.zeros((1, 1), dtype=jnp.int32), 100)[0].tolist()
print(decode(text))
