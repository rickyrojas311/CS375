import os
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class BarcodeDataset(Dataset):
    """
    A Dataset that yields (image, label) pairs for 32-bit unsigned integers.
    Each integer is converted to a 224x224x3 “barcode” image with 32 vertical
    stripes (each stripe is 7 pixels wide and 224 pixels tall).
    
    - 'split': either 'train' or 'val'
    - 'length': how many samples per epoch
    - 'val_numbers': a list of unique integers used ONLY for validation
    - 'seed': to control reproducibility (only used for training)
    """
    def __init__(self, split: str, length: int, val_numbers: list, seed: int = 1110):
        super().__init__()
        assert split in ['train', 'val']
        self.split = split
        self.length = length
        self.val_numbers = val_numbers
        
        # For reproducible random sampling
        self.rng = random.Random(seed if split == 'train' else seed + 9999)

        # Convert val_numbers to a set for quick membership checks
        self.val_set = set(val_numbers)
        self.max_uint32 = 2**32  # 0 to 2^32-1

        # For val split, we'll store val_numbers as a list
        if self.split == 'val':
            self.val_numbers_list = list(self.val_numbers)
            self.rng.shuffle(self.val_numbers_list)

    def __len__(self):
        # We define the 'epoch size' as self.length
        return self.length

    def __getitem__(self, index):
        if self.split == 'train':
            # Randomly pick a number not in the validation set
            while True:
                candidate = self.rng.randrange(0, self.max_uint32)
                if candidate not in self.val_set:
                    number = candidate
                    break
        else:
            # Validation: pick from val_numbers_list (cycling if needed)
            number = self.val_numbers_list[index % len(self.val_numbers_list)]

        # Build the (image, label) pair
        img_tensor = self.make_barcode_data(number)      # shape: [3, 224, 224]
        label_tensor = self.int_to_32bit_label(number)     # shape: [32]
        return img_tensor, label_tensor

    @staticmethod
    def make_barcode_data(number: int) -> torch.Tensor:
        """
        Given an unsigned 32-bit integer, produce a 224x224x3 “barcode” image.
        - 32 vertical stripes, each 7 pixels wide => total width = 224
        - Stripe i is black if bit i=1, white if bit i=0.
        """
        # Convert to a 32-bit binary array (bit 31 = LSB, bit 0 = MSB)
        bits = [(number >> i) & 1 for i in range(32)]
        bits.reverse()  # Now bits[0] is the MSB, bits[31] is the LSB

        # Create blank white image [H=224, W=224, C=3]
        img = np.ones((224, 224, 3), dtype=np.float32)  # 1.0 = white

        # Fill stripes for bits=1 with black (0.0)
        for i, bit in enumerate(bits):
            col_start = i * 7
            col_end = col_start + 7
            if bit == 1:
                img[:, col_start:col_end, :] = 0.0

        # Convert to torch tensor, shape [3, 224, 224]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        return img_tensor

    @staticmethod
    def int_to_32bit_label(number: int) -> torch.Tensor:
        """
        Convert the integer into a 32-bit binary label (0/1) (float32).
        bits[0] = MSB, bits[31] = LSB.
        """
        bits = [(number >> i) & 1 for i in range(32)]
        bits.reverse()
        return torch.tensor(bits, dtype=torch.float32)