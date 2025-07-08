import os
import tensorflow as tf

from model import build_mnist_cnn
from dataset import get_dataset
from EpochTracker import EpochTracker

train_dir = '../dataset/train'
val_dir = '../dataset/val'
model_dir = '../training_model'
os.makedirs(model_dir, exist_ok=True)

batch_size = 128
img_size = (28, 28)

epochs = 10
initial_epoch = 0
epoch_path = './last_epoch.txt'
if os.path.exists(epoch_path):
    with open(epoch_path, 'r') as f:
        initial_epoch = int(f.read())
    print(f"🔁 Continue training starting with epoch {initial_epoch + 1}")

train_ds, val_ds = get_dataset(train_dir, batch_size=batch_size, img_size=img_size)

model = None
last_model_path = os.path.join(model_dir, 'last.keras')
if os.path.exists(last_model_path):
    print("🔁 Load last saved model...")
    model = tf.keras.models.load_model(last_model_path)
else:
    print("🆕 New model created...")
    model = build_mnist_cnn(input_shape=(28, 28, 1), num_classes=10)

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Callbacks ===
checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(model_dir, 'best.keras'),
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)
checkpoint_last = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(model_dir, 'last.keras'),
    save_best_only=False,
    verbose=0
)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    factor=0.1,
    patience=3,
    min_delta=0.01,  # threshold
    verbose=1
)
epoch_tracker = EpochTracker(epoch_path)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    initial_epoch=initial_epoch,
    callbacks=[checkpoint_best, checkpoint_last, lr_scheduler, epoch_tracker]
)

# model.save(os.path.join(model_dir, 'final_model.keras'))