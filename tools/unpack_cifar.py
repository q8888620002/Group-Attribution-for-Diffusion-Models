import os
import pickle
import numpy as np
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_images_from_batch(data_batch, folder , batch_file):
    for i in range(len(data_batch[b'data'])):
        # Reshape the numpy array to a 3D array (32x32x3)
        image_array = data_batch[b'data'][i].reshape(3, 32, 32).transpose(1, 2, 0)
        
        # Convert to an Image object
        img = Image.fromarray(image_array)
        
        # Save image
        img.save(os.path.join(batch_path, folder, f'{batch_file}_image_{i}.png'))

batch_path = '/projects/leelab/mingyulu/data_att/cifar/cifar-10-batches-py' 
output_folder = 'cifar_images'  

os.makedirs(os.path.join(batch_path, output_folder), exist_ok=True)

# Load and process each batch
for batch_file in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']:
    batch_data = unpickle(os.path.join(batch_path, batch_file))
    save_images_from_batch(batch_data, output_folder, batch_file)