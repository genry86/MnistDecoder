import os
import shutil
import pandas as pd
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, random_split

class MnistDataset(Dataset):
    def __init__(self, path:str, transform=None):
        self.path = path
        self.transform = transform

        self.len_dataset = 0
        self.data_list = []

        for path_dir, dir_list, file_lists in os.walk(path):
            if path_dir == path:
                self.classes = sorted(dir_list)
                self.class_to_index = {class_name: i for i, class_name in enumerate(self.classes)}  # ["1":1, "2":2]
                continue

            cls = path_dir.split('/')[-1]   # class name

            for file_name in file_lists:
                file_path = os.path.join(path_dir, file_name)
                self.data_list.append((file_path, self.class_to_index[cls]))

            self.len_dataset += len(file_lists)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_path, hot_index = self.data_list[idx]
        sample = Image.open(file_path)

        if self.transform is not None:
            sample = self.transform(sample)
        # sample = np.array(sample)

        return sample, hot_index

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    training_data = MnistDataset("./dataset/train", transform=transform)
    test_data = MnistDataset("./dataset/val", transform=transform)

    # for path_dir, dir_list, file_lists in os.walk("./dataset/train"):
    #     print(f"Class - {path_dir.split('/')[-1]}")
    #     print(f"Dirs number - {len(dir_list)}")
    #     print(f"Files number - {len(file_lists)}")

    print("Classes:", training_data.classes)
    print("Classes To Index:", training_data.class_to_index)
    print(f"Len Training Data: {len(training_data)}")
    print(f"Len Testing Data: {len(test_data)}")

    print("items - ", training_data.class_to_index.items())

    for cls, position in training_data.class_to_index.items():
        vector = [int(i == position) for i in range(10)]
        print(f"{cls}: {vector}")

    train_data, val_data = random_split(training_data, [0.8, 0.2])
    print(f"train - {len(train_data)}")
    print(f"val - {len(val_data)}")

    batch_size = 128
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    train_loop = tqdm(train_loader, desc=f"Training - Epoch {1}/{1}", leave=True)
    for images, targets in train_loop:
        print(images)
        print(targets)

    # for index, (samples, labels) in enumerate(train_loader):
    #     if index < 2 or index == len(train_loader)-1:
    #         print(f"batch index - {index+1}")
    #         print(f"Samples - {samples.shape}")
    #         print(f"labels - {labels.shape}")

    print("******")
    rgb_img = np.array(Image.open("./image-bbf029d.jpg"))
    print(rgb_img.shape)                            # H, W, C

    print("******")
    img, hot_index = training_data[4888]
    cls = training_data.classes[hot_index]
    print(f"cls - {cls}")
    # transforms = transforms.ToTensor()
    transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    modified_img = transforms(img)
    print(type(modified_img))
    print(modified_img.shape)                   # C, H, W
    print(modified_img.dtype)
    print( f"min - {modified_img.min()}, max - {modified_img.max()}")

    # plt.imshow(img, cmap='gray')
    # plt.show()