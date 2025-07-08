import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# pytorch
import torch
from torchvision import transforms
from pt.model import MnistCNNModel

# rensorflow
import tensorflow as tf

RESULT_DIR = "training_model"
TEST_DATA_DIR = os.path.join("dataset", "val")

labels = sorted(os.listdir(TEST_DATA_DIR))
random_label = random.choice(labels)

TEST_LABEL_DIR = os.path.join(TEST_DATA_DIR, random_label)
images = sorted(os.listdir(TEST_LABEL_DIR))
random_image = random.choice(images)
TEST_IMAGE_PATH = os.path.join(TEST_LABEL_DIR, random_image)

img = Image.open(TEST_IMAGE_PATH).convert('L')
img_array = np.array(img, dtype=np.float32)

transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0)

model = MnistCNNModel(in_channels=1, out=10)
model_path_pt = os.path.join(RESULT_DIR, "best.pth")
model.load_state_dict(torch.load(model_path_pt))
output = model(img_tensor)
result = torch.argmax(output)
pytorch_result =  result.item()

model_path_tf = os.path.join(RESULT_DIR, "best.keras")
model = tf.keras.models.load_model(model_path_tf)
tf_img_array = np.expand_dims(img_array, 0)
tf_img_array = np.expand_dims(tf_img_array, -1)
output = model.predict(tf_img_array)
tf_resault = np.squeeze(output)
tf_resault = np.argmax(tf_resault)

plt.imshow(img_array, cmap='gray')
plt.title(f"Pytorch: {result.item()} | Tensorflow: {tf_resault}")
plt.show()