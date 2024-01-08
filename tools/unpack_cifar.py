import os
import pickle
from PIL import Image

def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

def process_batch(batch_data, batch_path, batch_file, num_classes=10):
    output_folders = [os.path.join(batch_path, f"cifar_images/oracle_excluded_{i}") for i in range(num_classes)]
    output_folders.append(os.path.join(batch_path, "cifar_images/all_labels"))

    for folder in output_folders:
        os.makedirs(folder, exist_ok=True)

    for i, label in enumerate(batch_data[b"labels"]):
        image_array = batch_data[b"data"][i].reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(image_array)

        for class_idx, output_folder in enumerate(output_folders):
            if class_idx == num_classes or class_idx != label:
                label_dir = os.path.join(output_folder, str(label))
                os.makedirs(label_dir, exist_ok=True)
                img.save(os.path.join(label_dir, f"{batch_file}_image_{i}.png"))

batch_path = "/gscratch/aims/datasets/cifar/cifar-10-batches-py"
batch_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

for batch_file in batch_files:
    batch_data = unpickle(os.path.join(batch_path, batch_file))
    process_batch(batch_data, batch_path, batch_file)
