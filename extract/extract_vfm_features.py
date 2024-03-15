import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import tqdm
import h5py
import torch
from torch.utils.data import DataLoader
from dataloader_video import VideoCLIPDataset

import math
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from transformers import CLIPImageProcessor

from tlvfm import vfm

data_folder = os.path.join("/mnt/hdd0/Charades_v1_480")

# Load VFM model.
config_dir = os.path.join("/video-foundation-model/tlvfm_configs/umt-official-vitl16-multimodal.yaml")
ckpt_name = "umt-official-pretrained-vitl16-multimodal_25m.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = vfm(cfg_dir=vision_tower_cfg.video_tower_cfg_dir,
            ckpt_name=vision_tower_cfg.video_tower_ckpt_name,
            ckpt_dir="../checkpoints",
            quiet=True)
if torch.cuda.is_available():
    model = model.cuda()
feat_folder = os.path.join(data_folder, "features")
os.makedirs(feat_folder, exist_ok=True)

processor = CLIPImageProcessor(image_mean=vision_model.get_image_mean(), image_std=vision_model.get_image_std())

frame_num = model.get_input_shape_info()['num_frames']
dataset = VideoCLIPDataset(None, frame_num, os.path.join(data_folder, "*.mp4"))
print(len(dataset))

dataloader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=16,
    shuffle=False
)
data_iter = iter(dataloader)

dataset_feats = h5py.File(os.path.join(data_folder, "{}.h5".format(ckpt_name.split('.')[0])), "w")
G = self.vision_model.get_input_shape_info()['grid_size'][0]
D = model.get_output_shape_info()["patch"][-1]
dataset_feats.create_dataset("features", (len(dataset), frame_num, G * G, D))
dataset_feats.create_dataset("ids", (len(dataset), ), 'S20')

global_index = 0
video_ids = {}
data_iter = iter(dataloader)
for batch in tqdm.tqdm(data_iter):
    batch_size = batch['video'].shape[0]
    for i in range(batch_size):
        with torch.no_grad():  
            video_features = model(batch['video'][i].cuda())
        dataset_feats['features'][global_index] = image_features.detach().cpu().numpy()
        dataset_feats['ids'][global_index] = batch['vid'][i].encode("ascii", "ignore")  
        global_index += 1
        
dataset_feats.close()