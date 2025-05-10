from data_utils import download_dataset

# Download and prepare training datasets
train_clean_100_path = download_dataset("luizfelipebjcosta/libri-tts-train-clean-100")
train_clean_360_path = download_dataset("luizfelipebjcosta/libri-tts-train-clean-360-part-1")
musan_noise_path = download_dataset("nhattruongdev/musan-noise")

print("Training dataset paths:")
print("train-clean-100:", train_clean_100_path)
print("train-clean-360:", train_clean_360_path)
print("musan-noise:", musan_noise_path)
