import tensorflow as tf
import argparse
import os
import shutil
import numpy as np
import tempfile


def register_custom_layers():
    """Register custom layers used in the VAD model."""
    class ResizeToInputFrames(tf.keras.layers.Layer):
        def __init__(self, target_time_frames, **kwargs):
            super(ResizeToInputFrames, self).__init__(**kwargs)
            self.target_time_frames = target_time_frames

        def call(self, inputs):
            current_time_frames = tf.shape(inputs)[1]

            def resize_fn():
                scale = tf.cast(current_time_frames, tf.float32) / \
                    tf.cast(self.target_time_frames, tf.float32)
                indices = tf.range(self.target_time_frames,
                                   dtype=tf.float32) * scale
                indices = tf.cast(tf.floor(indices), tf.int32)
                indices = tf.clip_by_value(indices, 0, current_time_frames - 1)
                return tf.gather(inputs, indices, axis=1)

            def identity_fn():
                return inputs

            should_resize = tf.not_equal(
                current_time_frames, self.target_time_frames)
            return tf.cond(should_resize, resize_fn, identity_fn)

        def get_config(self):
            config = super(ResizeToInputFrames, self).get_config()
            config.update({'target_time_frames': self.target_time_frames})
            return config

    return {"ResizeToInputFrames": ResizeToInputFrames}


def recreate_vad_model(input_shape=(99, 40, 1)):
    """
    Recreate the VAD model from scratch with frozen BatchNormalization layers.
    This approach can avoid TFLite conversion issues by ensuring BatchNormalization
    layers are properly configured from the start.
    """
    print("🏗️ Recreating VAD model with frozen batch normalization...")

    tf.keras.backend.clear_session()

    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)

    # CNN for feature extraction
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization(trainable=False)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)  # (49, 20, 16)

    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(trainable=False)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(x)  # (49, 10, 32)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(trainable=False)(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Reshape for further processing
    x = tf.keras.layers.Reshape(
        (x.shape[1], x.shape[2] * x.shape[3]))(x)  # (49, 10*64)

    # Temporal modeling with dilated convolutions
    x = tf.keras.layers.Conv1D(
        64, kernel_size=1, padding='same', activation='relu')(x)

    # Dilated convolutions to capture temporal dependencies
    x = tf.keras.layers.Conv1D(64, kernel_size=3, dilation_rate=1,
                               padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(trainable=False)(x)

    x = tf.keras.layers.Conv1D(64, kernel_size=3, dilation_rate=2,
                               padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(trainable=False)(x)

    x = tf.keras.layers.Conv1D(64, kernel_size=3, dilation_rate=4,
                               padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(trainable=False)(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    # Frame-level predictions
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1, activation='sigmoid'))(x)
    outputs = tf.keras.layers.Reshape((x.shape[1],))(
        outputs)  # Flatten to (time_frames,)

    # Create the clean model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    print("✅ Clean model recreation complete")

    return model


def transfer_weights(source_model, target_model):
    """
    Transfer weights from the source model to the target model,
    mapping the layers by name.
    """
    print("🔄 Transferring weights from source model to target model...")

    # Build a dictionary of source model weights by layer name
    source_weights = {}
    for layer in source_model.layers:
        if layer.weights:
            source_weights[layer.name] = layer.get_weights()

    # Apply weights to matching layers in target model
    transferred_count = 0
    for layer in target_model.layers:
        if layer.name in source_weights:
            try:
                layer.set_weights(source_weights[layer.name])
                print(f"  ✓ Transferred weights for layer: {layer.name}")
                transferred_count += 1
            except ValueError as e:
                print(
                    f"  ✗ Failed to transfer weights for layer: {layer.name} - {e}")

    print(f"✅ Weight transfer complete: {transferred_count} layers updated")
    return target_model


def convert_to_tflite_with_workaround(model_path, output_path, quantize=False):
    """
    Convert the VAD model to TFLite format using a clean model approach
    to avoid batch normalization issues.
    """
    print(f"📋 Loading original model from {model_path}...")

    custom_objects = register_custom_layers()

    try:
        # Load the original model
        tf.keras.config.enable_unsafe_deserialization()
        original_model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        print("✅ Original model loaded successfully")
    except Exception as e:
        print(f"⚠️ Model loading failed: {e}")
        raise ValueError(
            "Could not load the original model. Check the format and dependencies.")

    # Get the model's architecture
    input_shape = original_model.input_shape[1:]  # Remove batch dimension
    print(f"📊 Model input shape: {input_shape}")

    # Create a clean version of the model with frozen BatchNormalization
    clean_model = recreate_vad_model(input_shape)

    # Transfer weights from original model to clean model
    clean_model = transfer_weights(original_model, clean_model)

    # Ensure all BatchNormalization layers are properly frozen
    for layer in clean_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Run a forward pass to ensure BN layers use moving statistics
    test_input = np.zeros((1,) + input_shape)
    _ = clean_model(test_input, training=False)

    # Create a temporary directory for SavedModel
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Use export() to get a SavedModel
        print(f"📤 Exporting model to SavedModel format at: {tmp_dir}")
        clean_model.export(tmp_dir)

        # Convert to TFLite
        print("🔄 Converting SavedModel to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)

        # Configure conversion parameters
        if quantize:
            print("🔧 Applying quantization optimizations...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Essential options for handling complex ops
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]

        try:
            tflite_model = converter.convert()
            with open(output_path, 'wb') as f:
                f.write(tflite_model)

            model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(
                f"✅ TFLite model saved to {output_path} (Size: {model_size_mb:.2f} MB)")
            return True
        except Exception as e:
            print(f"❌ TFLite conversion failed: {e}")

            # Try alternative approach
            try:
                print("🔄 Trying alternative conversion method...")

                # Save model to Keras format
                temp_keras_path = f"{os.path.splitext(output_path)[0]}_temp.keras"
                clean_model.save(temp_keras_path)
                print(f"💾 Saved intermediate Keras model to {temp_keras_path}")

                # Convert using direct TFLite converter
                converter = tf.lite.TFLiteConverter.from_keras_model(
                    clean_model)

                if quantize:
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]

                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS]

                tflite_model = converter.convert()
                with open(output_path, 'wb') as f:
                    f.write(tflite_model)

                # Clean up
                if os.path.exists(temp_keras_path):
                    os.remove(temp_keras_path)

                model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(
                    f"✅ TFLite model saved using alternative method to {output_path} (Size: {model_size_mb:.2f} MB)")
                return True
            except Exception as e2:
                print(f"❌ Alternative conversion also failed: {e2}")

                # Try with another method (last resort)
                try:
                    print("🔄 Trying final fallback method...")

                    # Create a concrete function
                    print("📋 Creating concrete function...")
                    concrete_func = tf.function(lambda x: clean_model(x))
                    concrete_func = concrete_func.get_concrete_function(
                        tf.TensorSpec([1] + list(input_shape), tf.float32))

                    # Convert from concrete function
                    converter = tf.lite.TFLiteConverter.from_concrete_functions([
                                                                                concrete_func])

                    if quantize:
                        converter.optimizations = [tf.lite.Optimize.DEFAULT]

                    converter.target_spec.supported_ops = [
                        tf.lite.OpsSet.TFLITE_BUILTINS]

                    tflite_model = converter.convert()
                    with open(output_path, 'wb') as f:
                        f.write(tflite_model)

                    model_size_mb = os.path.getsize(
                        output_path) / (1024 * 1024)
                    print(
                        f"✅ TFLite model saved using final fallback method to {output_path} (Size: {model_size_mb:.2f} MB)")
                    return True
                except Exception as e3:
                    print(
                        f"❌ All conversion methods failed. Final error: {e3}")
                    return False


def verify_tflite_model(tflite_path, input_shape):
    """Verify the converted TFLite model with test input."""
    print(f"🔍 Verifying TFLite model at {tflite_path}...")

    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"📊 Input tensor details: {input_details}")
        print(f"📊 Output tensor details: {output_details}")

        # Create sample input
        sample_input = np.random.random(input_shape).astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], sample_input)
        interpreter.invoke()

        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])

        print(f"✅ TFLite model verification successful")
        print(f"📊 Output shape: {output.shape}")
        print(f"📊 Output range: [{np.min(output)}, {np.max(output)}]")

        return True
    except Exception as e:
        print(f"❌ TFLite model verification failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert VAD model to TFLite with batch normalization fix")
    parser.add_argument("--model_path", required=True,
                        help="Path to the trained VAD model file")
    parser.add_argument("--output_path", default="vad_model_fixed.tflite",
                        help="Output TFLite model path")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply post-training quantization")
    parser.add_argument("--verify", action="store_true",
                        help="Verify the converted TFLite model")

    args = parser.parse_args()

    # Convert the model with workaround
    success = convert_to_tflite_with_workaround(
        model_path=args.model_path,
        output_path=args.output_path,
        quantize=args.quantize
    )

    if success and args.verify:
        # Verify the converted model
        input_shape = (1, 99, 40, 1)  # Default VAD model input shape
        verify_tflite_model(args.output_path, input_shape)

    if success:
        print("✅ Conversion completed successfully")
    else:
        print("❌ Conversion failed")
        exit(1)
