import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

hot_vectors = np.eye(10)

def load_paths_and_labels(root_dir):
    image_paths, labels = [], []

    for label_name in sorted(os.listdir(root_dir)):
        label_path = os.path.join(root_dir, label_name)
        if not os.path.isdir(label_path):
            continue

        index = int(label_name)  # label = name of the folder
        label = hot_vectors[index]

        for fname in os.listdir(label_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(label_path, fname))
                labels.append(label)
    return image_paths, labels

def create_tf_dataset(image_paths, labels, img_size=(28, 28), batch_size=128, shuffle=True):
    image_paths = tf.convert_to_tensor(image_paths)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(len(image_paths))

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

def get_dataset(image_dir, batch_size=128, img_size=(28, 28)):
    image_paths, labels = load_paths_and_labels(image_dir)

    # Split train/val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.1, random_state=42
    )

    train_ds = create_tf_dataset(train_paths, train_labels, batch_size=batch_size, img_size=img_size)
    val_ds = create_tf_dataset(val_paths, val_labels, shuffle=False)

    return train_ds, val_ds

if __name__ == '__main__':
    root_dir = '../dataset/train'
    batch_size = 128
    img_size = (28, 28)

    train_ds, val_ds = get_dataset(root_dir, img_size=img_size, batch_size=batch_size)
    print(len(train_ds), len(val_ds))

    for i, (images, labels) in enumerate(train_ds.take(5)):
        print(f"[{i}] Images shape: {images.shape}, Labels shape: {labels.shape}")
        # [0] Images shape: (128, 28, 28, 1), Labels shape: (128, 10)
        # [1] Images shape: (128, 28, 28, 1), Labels shape: (128, 10)
        # [2] Images shape: (128, 28, 28, 1), Labels shape: (128, 10)
        # [3] Images shape: (128, 28, 28, 1), Labels shape: (128, 10)
        # [4] Images shape: (128, 28, 28, 1), Labels shape: (128, 10)

# class MNISTSequence(Sequence):
#     def __init__(self, root_dir, batch_size=64, img_size=(28, 28), shuffle=True):
#         self.root_dir = root_dir
#         self.batch_size = batch_size
#         self.img_size = img_size
#         self.shuffle = shuffle
#
#         self.samples = []  # список (путь_к_файлу, метка)
#         for label in sorted(os.listdir(root_dir)):
#             class_dir = os.path.join(root_dir, label)
#             if os.path.isdir(class_dir):
#                 for fname in os.listdir(class_dir):
#                     self.samples.append((os.path.join(class_dir, fname), int(label)))
#
#         self.on_epoch_end()
#
#     def __len__(self):
#         return len(self.samples) // self.batch_size
#
#     def __getitem__(self, index):
#         batch_samples = self.samples[index * self.batch_size:(index + 1) * self.batch_size]
#         X, y = [], []
#
#         for path, label in batch_samples:
#             img = load_img(path, target_size=self.img_size, color_mode='grayscale')
#             img = img_to_array(img) / 255.0  # нормализация [0, 1]
#             X.append(img)
#             y.append(label)
#
#         return np.array(X), np.array(y)
#
#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.samples)

# def get_data_generators(train_dir, val_dir, img_size=(28, 28), batch_size=64):
#     """
#     Создаёт генераторы для загрузки изображений из папок
#     (ожидается структура: dataset/train/0, ..., dataset/train/9)
#     """
#     # Препроцессинг: нормализация значений пикселей
#     train_datagen = ImageDataGenerator(rescale=1./255)
#     val_datagen = ImageDataGenerator(rescale=1./255)
#
#     train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=img_size,
#         color_mode='grayscale',
#         batch_size=batch_size,
#         class_mode='sparse',  # для метки — числа от 0 до 9
#         shuffle=True
#     )
#
#     val_generator = val_datagen.flow_from_directory(
#         val_dir,
#         target_size=img_size,
#         color_mode='grayscale',
#         batch_size=batch_size,
#         class_mode='sparse',
#         shuffle=False
#     )
#
#     return train_generator, val_generator