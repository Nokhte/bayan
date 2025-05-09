import tensorflow as tf
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, Conv1D, Dense,
    BatchNormalization, Activation,
    Reshape, TimeDistributed,  Layer
)


class ResizeToInputFrames(tf.keras.layers.Layer):
    def __init__(self, target_time_frames, **kwargs):
        super(ResizeToInputFrames, self).__init__(**kwargs)
        self.target_time_frames = target_time_frames

    def call(self, inputs):
        # Input shape: (batch, time_frames)
        current_time_frames = tf.shape(inputs)[1]

        # Use tf.cond to handle conditional logic in graph mode
        def resize_fn():
            # Compute interpolation indices
            scale = tf.cast(current_time_frames, tf.float32) / \
                tf.cast(self.target_time_frames, tf.float32)
            indices = tf.range(self.target_time_frames,
                               dtype=tf.float32) * scale
            indices = tf.cast(tf.floor(indices), tf.int32)
            indices = tf.clip_by_value(indices, 0, current_time_frames - 1)
            return tf.gather(inputs, indices, axis=1)

        def identity_fn():
            return inputs

        # Check if resizing is needed using tf.equal
        should_resize = tf.not_equal(
            current_time_frames, self.target_time_frames)
        return tf.cond(should_resize, resize_fn, identity_fn)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.target_time_frames)

    def get_config(self):
        config = super(ResizeToInputFrames, self).get_config()
        config.update({'target_time_frames': self.target_time_frames})
        return config


def create_vad_model(input_shape=(99, 40, 1)):
    """
    Create a Voice Activity Detection model that's efficient for real-time processing.
    The model uses a CNN-RNN architecture for effective temporal feature extraction.

    Args:
        input_shape: Shape of input mel spectrogram (time_frames, mel_bins, channels)

    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = Input(shape=input_shape)

    # CNN for feature extraction
    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # (49, 20, 16)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)  # (49, 10, 32)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Reshape for RNN processing
    x = Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)  # (49, 10*64)

    # Bidirectional GRU for temporal modeling
    # Instead of bidirectional GRU, use a stack of 1D convolutions with increasing dilation
    # This gives a large receptive field similar to RNNs
    x = Conv1D(64, kernel_size=1, padding='same', activation='relu')(x)

    # First dilated convolution - captures short-term dependencies
    x = Conv1D(64, kernel_size=3, dilation_rate=1,
               padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Second dilated convolution - captures medium-term dependencies
    x = Conv1D(64, kernel_size=3, dilation_rate=2,
               padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Third dilated convolution - captures longer-term dependencies
    x = Conv1D(64, kernel_size=3, dilation_rate=4,
               padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Dropout(0.3)(x)

    # Frame-level predictions
    outputs = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    outputs = Reshape((x.shape[1],))(outputs)  # Flatten to (time_frames,)

    # Ensure output matches expected dimensions
    assert outputs.shape[1] == 49, f"Expected 49 output frames but got {outputs.shape[1]}"

    # Resize to match input time frames
    if input_shape[0] != outputs.shape[1]:
        outputs = ResizeToInputFrames(
            target_time_frames=input_shape[0])(outputs)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model


if __name__ == "__main__":
    # Test the model and print summary
    model = create_vad_model()
    model.summary()

    # Calculate model size
    trainable_count = sum(tf.keras.backend.count_params(w)
                          for w in model.trainable_weights)
    non_trainable_count = sum(tf.keras.backend.count_params(w)
                              for w in model.non_trainable_weights)

    print(f"Total parameters: {trainable_count + non_trainable_count}")
    print(f"Trainable parameters: {trainable_count}")
    print(f"Non-trainable parameters: {non_trainable_count}")

    # Create a test batch and ensure the model works
    import numpy as np
    test_batch = np.random.random((1, 99, 40, 1))
    predictions = model.predict(test_batch)

    print(f"Input shape: {test_batch.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Output range: [{np.min(predictions)}, {np.max(predictions)}]")
