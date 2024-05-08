"""Unpack CIFAR-10 data into png image files."""
import os
import pickle
import random

from PIL import Image
import torchvision.transforms as transforms

from torchvision.datasets import CIFAR100
from src.datasets import  CIFAR100_filter

def unpickle(file):
    """Unpickle batch data saved with bytes encoding."""
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def process_batch(batch_data, output_path, batch_file, labels=None):
    """Process a batch in CIFAR-10."""
    if batch_file != "test_batch":
        print(f"Processing training batch: {batch_file}")
        output_folder = os.path.join(output_path, "cifar_images", "train")
    else:
        print(f"Processing test batch: {batch_file}")
        output_folder = os.path.join(output_path, "cifar_images", "test")

    os.makedirs(output_folder, exist_ok=True)

    if labels is not None:
        for i, (image_array, label) in enumerate(
            zip(batch_data[b"data"], batch_data[b"labels"])
        ):
            if label in labels:
                image_array = image_array.reshape(3, 32, 32).transpose(1, 2, 0)
                img = Image.fromarray(image_array)
                img.save(os.path.join(output_folder, f"{batch_file}_image_{i}.png"))
    else:
        for i, image_array in enumerate(batch_data[b"data"]):
            image_array = image_array.reshape(3, 32, 32).transpose(1, 2, 0)
            img = Image.fromarray(image_array)
            img.save(os.path.join(output_folder, f"{batch_file}_image_{i}.png"))


def save_images_with_labels(dataset, save_path, target_labels=None):

    # Loop through the dataset
    os.makedirs(save_path, exist_ok=True)

    for i in range(len(dataset)):
        image, label = dataset[i]
        
        if target_labels is not None:
            if label in target_labels:            
                image.save(os.path.join(save_path, f'img_{i}_label_{label}.jpg'))
        else:
            image.save(os.path.join(save_path, f'img_{i}_label_{label}.jpg'))

trainset = CIFAR100(root='/gscratch/aims/datasets/cifar100', train=True, download=True)

random.seed(42)
target_labels = random.sample(range(100), 25)
save_images_with_labels(trainset, '/gscratch/aims/datasets/cifar100_f/filtered_2', target_labels)


# classes_to_keep = [
#             # 0, 1, 2, 3, 4,   # Aquatic mammals
#             # 5, 6, 7, 8, 9,   # Fish
#             # 75, 76, 77, 78, 79,  # Reptiles

#             40, 41, 42, 43, 44,  # Large carnivores
#             55, 56, 57, 58, 59,  # Large omnivores and herbivores
#             60, 61, 62, 63, 64,  # Medium mammals
#             80, 81, 82, 83, 84   # Small mammals
# ]

# batch_path = "/gscratch/aims/datasets/cifar/cifar-10-batches-py"
# batch_files = [
#     "data_batch_1",
#     "data_batch_2",
#     "data_batch_3",
#     "data_batch_4",
#     "data_batch_5",
#     "test_batch",
# ]

# labels = {1, 7}  # Filter for 'automobile' and 'horse' images

# output_folder = os.path.join(batch_path, "cifar2")
# os.makedirs(output_folder, exist_ok=True)

# for batch_file in batch_files:
#     batch_data = unpickle(os.path.join(batch_path, batch_file))
#     process_batch(batch_data, output_folder, batch_file, labels)
# print("Done!")
