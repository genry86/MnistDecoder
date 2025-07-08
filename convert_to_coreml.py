import os
import coremltools as ct

import torch
from pt.model import MnistCNNModel

import tensorflow as tf

MODEL_PATH = os.path.join("training_model", "best.pth")
dst = os.path.join("training_model", "mnist_pytorch.mlpackage")

model = MnistCNNModel(in_channels=1, out=10)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Trace
input_tensor = torch.rand(1, 1, 28, 28)
traced_model = torch.jit.trace(model, input_tensor)

coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="input_1", shape=input_tensor.shape)],
    convert_to="mlprogram"
)
coreml_model.save(dst)

MODEL_PATH = os.path.join("training_model", "best.keras")
dst = os.path.join("training_model", "mnist_tensorflow.mlpackage")

model = tf.keras.models.load_model(MODEL_PATH)
input_shape = (ct.RangeDim(1, 128), 28, 28, 1)  # ← dynamic batch size

coreml_model = ct.convert(
    model,
    inputs=[ct.TensorType(name="input_1", shape=input_shape)],
    convert_to="mlprogram",
    source='tensorflow'
)
coreml_model.save(dst)