import os
import argparse
import tensorflow as tf
import numpy as np
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
from datetime import datetime
from model import create_vad_model
from scripts.vad_dataset import create_tf_dataset

# Set memory growth for GPUs
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"Using {len(physical_devices)} GPU(s)")
else:
    print("No GPU found, using CPU")


def parse_args():
    parser = argparse.ArgumentParser(description='Train VAD model')
    parser.add_argument('--speech_dir', type=str, required=True,
                        help='Directory containing speech files')
    parser.add_argument('--noise_dir', type=str, required=True,
                        help='Directory containing noise files')
    parser.add_argument('--val_speech_dir', type=str,
                        help='Directory containing validation speech files (if different from training)')
    parser.add_argument('--val_noise_dir', type=str,
                        help='Directory containing validation noise files (if different from training)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save the model')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--snr_range', type=str, default='5,15',
                        help='SNR range in dB (min,max)')
    parser.add_argument('--val_steps', type=int, default=50,
                        help='Number of validation steps')
    parser.add_argument('--steps_per_epoch', type=int, default=500,
                        help='Steps per epoch')
    parser.add_argument('--tensorboard_dir', type=str, default='logs',
                        help='Directory for TensorBoard logs')
    return parser.parse_args()


def plot_training_history(history, model_dir):
    """Plot and save the training history"""
    plt.figure(figsize=(12, 8))

    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history.history['binary_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_binary_accuracy'],
             label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot precision
    plt.subplot(2, 2, 3)
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)

    # Plot recall
    plt.subplot(2, 2, 4)
    plt.plot(history.history['recall'], label='Training Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    plt.close()


def main():
    args = parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Parse SNR range
    snr_min, snr_max = map(float, args.snr_range.split(','))
    snr_range = (snr_min, snr_max)

    print(f"Creating training dataset with SNR range: {snr_range} dB")

    # Calculate the number of training examples needed
    train_examples_needed = args.steps_per_epoch * args.batch_size
    val_examples_needed = args.val_steps * args.batch_size

    print(
        f"Generating {train_examples_needed} training examples for {args.epochs} epochs")
    print(f"Generating {val_examples_needed} validation examples")

    # Create training dataset - make sure to create enough examples for all epochs
    train_dataset = create_tf_dataset(
        speech_dir=args.speech_dir,
        noise_dir=args.noise_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        snr_range=snr_range,
        num_examples=train_examples_needed,
    )

    # Create validation dataset
    val_speech_dir = args.val_speech_dir if args.val_speech_dir else args.speech_dir
    val_noise_dir = args.val_noise_dir if args.val_noise_dir else args.noise_dir

    val_dataset = create_tf_dataset(
        speech_dir=val_speech_dir,
        noise_dir=val_noise_dir,
        batch_size=args.batch_size,
        seed=args.seed + 1,  # Different seed for validation
        snr_range=snr_range,
        num_examples=val_examples_needed,

    )

    # Create model
    model = create_vad_model()
    model.summary()

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)

    # Set up callbacks
    log_dir = os.path.join(args.tensorboard_dir,
                           datetime.now().strftime("%Y%m%d-%H%M%S"))

    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(args.model_dir, 'vad_model_best.keras'),
            save_best_only=True,
            monitor='val_binary_accuracy',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='epoch'
        )
    ]

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=args.val_steps,
        callbacks=callbacks,
        verbose=1
    )

    # Save the final model
    final_model_path = os.path.join(args.model_dir, 'vad_model_final.keras')
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # # Convert to TFLite with error handling
    # try:
    #     # First, try with the normal conversion
    #     print("Converting model to TFLite format...")
    #     converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #     converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #     tflite_model = converter.convert()

    #     tflite_path = os.path.join(args.model_dir, 'vad_model.tflite')
    #     with open(tflite_path, 'wb') as f:
    #         f.write(tflite_model)
    #     print(f"TFLite model saved to {tflite_path}")
    # except Exception as e:
    #     print(f"Error during TFLite conversion: {e}")
    #     print("Trying alternative conversion method...")

    #     try:
    #         # Try with saved model path
    #         saved_model_path = os.path.join(args.model_dir, 'saved_model')
    #         tf.saved_model.save(model, saved_model_path)

    #         converter = tf.lite.TFLiteConverter.from_saved_model(
    #             saved_model_path)
    #         converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #         # Add more compatibility options
    #         converter.target_spec.supported_ops = [
    #             tf.lite.OpsSet.TFLITE_BUILTINS,
    #             tf.lite.OpsSet.SELECT_TF_OPS
    #         ]
    #         tflite_model = converter.convert()

    #         tflite_path = os.path.join(args.model_dir, 'vad_model.tflite')
    #         with open(tflite_path, 'wb') as f:
    #             f.write(tflite_model)
    #         print(f"TFLite model saved to {tflite_path}")
    #     except Exception as e2:
    #         print(f"Alternative TFLite conversion also failed: {e2}")
    #         print(
    #             "Skipping TFLite conversion. You can try converting the model manually later.")

    # Plot and save training history
    plot_training_history(history, args.model_dir)

    print("Training completed successfully!")
    print(f"Model saved to {args.model_dir}")


if __name__ == "__main__":
    main()
