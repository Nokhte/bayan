import tensorflow as tf
import os
import train_soroush

if os.path.isfile('pretrained/saved_model.pb'):
    # git push problem
    model = tf.keras.models.load_model('pretrained')
else:
    model = train_soroush.BaseSpeechEmbeddingModel()
    model.load_weights('pretrained/cp-0110.ckpt')
    # after "save", you can use load_model without problems
    model.save('pretrained')

# Load the SavedModel
saved_model = tf.saved_model.load("pretrained")

# Get the serving signature
infer = saved_model.signatures["serving_default"]

# Define the input signature with a fixed shape
input_signature = [tf.TensorSpec([1, 48000], tf.float32, name="input")]

# Create a tf.function with the specified input signature
@tf.function(input_signature=input_signature)
def wrapped_infer(input_tensor):
    return infer(input_tensor)


# Get the concrete function
concrete_func = wrapped_infer.get_concrete_function()

# Create the TFLite converter from the concrete function
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

# Configure the converter for compatibility
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_lower_tensor_list_ops = False
converter.experimental_enable_resource_variables = True

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open("out/soroush.tflite", "wb") as f:
    f.write(tflite_model)
