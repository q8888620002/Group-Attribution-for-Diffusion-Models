"""Unpack CIFAR-10 data into png image files."""
import os
import pickle

from PIL import Image


def unpickle(file):
    """Unpickle batch data saved with bytes encoding."""
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def process_batch(batch_data, batch_path, batch_file, num_classes=10):
    """Process a batch in CIFAR-10."""
    if batch_file != "test_batch":
        print(f"Processing training batch: {batch_file}")
        output_folder = os.path.join(batch_path, "cifar_images", "train")
    else:
        print(f"Processing test batch: {batch_file}")
        output_folder = os.path.join(batch_path, "cifar_images", "test")

    os.makedirs(output_folder, exist_ok=True)
    for i, image_array in enumerate(batch_data[b"data"]):
        image_array = image_array.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(image_array)
        img.save(os.path.join(output_folder, f"{batch_file}_image_{i}.png"))


# TODO: Remove line this for anonymized code submission.
batch_path = "/gscratch/aims/datasets/cifar/cifar-10-batches-py"
batch_files = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
    "test_batch",
]

for batch_file in batch_files:
    batch_data = unpickle(os.path.join(batch_path, batch_file))
    process_batch(batch_data, batch_path, batch_file)
print("Done!")
