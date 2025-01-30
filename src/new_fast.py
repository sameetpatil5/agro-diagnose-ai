import tensorflow as tf
import tf2onnx

# Load trained CNN model
MODEL_PATH = (
    "../models/tl_inceptionv3_raw_i224_b32_e50_2025_01_30_08_21_32_AM.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)

# Ensure the model is Functional (if originally Sequential)
if isinstance(model, tf.keras.Sequential):
    inputs = tf.keras.Input(shape=model.input_shape[1:])
    outputs = model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Check model input shape (important!)
print(
    f"✅ TensorFlow Model Input Shape: {model.input_shape}"
)  # Expected (None, 128, 128, 3)

# Define input signature explicitly for ONNX
input_signature = [
    tf.TensorSpec([None] + list(model.input_shape[1:]), tf.float32, name="input")
]

# Convert to ONNX (Explicitly setting channels-last format)
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    inputs_as_nchw=None,  # Keeps channels-last format (ONNX: NHWC)
)

# Save ONNX model
onnx_model_path = (
    "../models/tl_inceptionv3_raw_i224_b32_e50_ft2.onnx"
)

with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"✅ CNN Model successfully converted to ONNX: {onnx_model_path}")

# Verify ONNX input shape
import onnx

onnx_model = onnx.load(onnx_model_path)
onnx_inputs = onnx_model.graph.input
for inp in onnx_inputs:
    print(f"🔍 ONNX Model Input Shape: {inp.type.tensor_type.shape.dim}")
