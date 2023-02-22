import numpy as np
from torch.utils.data import Dataset, DataLoader


class MiniShakesPeare(Dataset):
    def __init__(self, data, block_size=8) -> None:
        super().__init__()
        self.block_size = block_size
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index == -1:
            index = len(self.data) - 1
        x = self.data[index : index + self.block_size]
        y = self.data[index + 1 : index + self.block_size + 1]

        if index + self.block_size + 1 > len(self.data):
            diff = index + self.block_size + 1 - len(self.data)

            to_add_on_x = diff - 1
            to_add_on_y = diff

            x = np.concatenate((x, self.data[:to_add_on_x]))
            y = np.concatenate((y, self.data[:to_add_on_y]))

        return x, y


def get_data(batch_size=4, train_ratio=0.9, block_size=8):
    with open("input.txt", "r") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    # print("".join(chars))
    vocabulary_size = len(chars)

    # Lookup table to map single characters to integers
    char_to_idx = {ch: i for i, ch in enumerate(chars)}

    # Lookup table to map integers to single characters
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    encode = lambda string: np.array([char_to_idx[ch] for ch in string])
    decode = lambda latent: "".join([idx_to_char[idx] for idx in latent])
    data = np.array(encode(text))
    n = int(train_ratio * len(data))

    train_data = data[:n]
    test_data = data[n:]

    train_dataset = MiniShakesPeare(train_data, block_size=block_size)

    test_dataset = MiniShakesPeare(test_data, block_size=block_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_dataloader, test_dataloader, vocabulary_size, encode, decode
