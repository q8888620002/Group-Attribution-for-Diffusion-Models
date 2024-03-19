"""Converting images to BLIP embedding"""
import os
import torch
import requests
import json

from src.constants import DATASET_DIR
from PIL import Image
from transformers import BlipImageProcessor, BlipVisionModel

clusters_12 = json.load(open(os.path.join(DATASET_DIR, "clusters/clusters_id_all_blip_clusters_12.json")))
clusters_24 = json.load(open(os.path.join(DATASET_DIR, "clusters/clusters_id_all_blip_clusters_24.json")))
clusters_48 = json.load(open(os.path.join(DATASET_DIR, "clusters/clusters_id_all_blip_clusters_48.json")))

clusters_by_size = {
    12: clusters_12,
    24: clusters_24,
    48: clusters_48,
}

# centroid

processor = BlipImageProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipVisionModel.from_pretrained("Salesforce/blip-vqa-base")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')


inputs = processor(images=image, return_tensors="pt")
image_features = model(**inputs).pooler_output

import ipdb;ipdb.set_trace()