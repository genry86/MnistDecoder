import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

def save_mnist_images(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    # create folders (0-9)
    for label in sorted(df['label'].unique()):
        class_dir = os.path.join(output_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)

    # Handle each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(csv_path)}"):
        label = row['label']
        pixels = row.iloc[1:].values.astype(np.uint8)
        image = pixels.reshape((28, 28))
        img = Image.fromarray(image, mode='L')
        img.save(os.path.join(output_dir, str(label), f"{idx}.png"))

train_csv = "data/mnist_train.csv"
test_csv = "data/mnist_test.csv"
train_output_dir = "dataset/train"
val_output_dir = "dataset/val"

save_mnist_images(train_csv, train_output_dir)
save_mnist_images(test_csv, val_output_dir)