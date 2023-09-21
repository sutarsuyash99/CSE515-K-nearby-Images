from torchvision.datasets import Caltech101
from color_moments import rgb_color_moments
from resnet_50 import resnet_50_init
from hog import compute_hog
import torch
from tqdm import tqdm

def bulk_hog(dataset, new_size: tuple):
    print("-"*15, "Starting Hog Descriptors", "-"*15)
    pkl_hog = 'hog.pkl'
    map = {}
    for i in tqdm(range(len(dataset))):
        img, _ = dataset[i]
        features = compute_hog(pil_img=img, new_size=(300, 100), in_bulk=True)
        map[i] = features
    
    torch.save(map, pkl_hog)

def bulk_color_moments(dataset, new_size: tuple):
    print("-"*15, "Starting Color Moments", "-"*15)
    pik = "color_moments.pkl"
    map = {}
    for i in tqdm(range(len(dataset))):
        img, _ = dataset[i]
        features = rgb_color_moments(pil_img=img, new_size=new_size, in_bulk=True)
        map[i] = features

    torch.save(map, pik)

def bulk_resnet(dataset: Caltech101, new_size: tuple):
    print("-"*15, "Starting Resnets", "-"*15)
    pkl_avgpool, pkl_layer3, pkl_fc = 'resnet_avgpool.pkl', 'resnet_layer3.pkl', 'resnet_fc.pkl'
    map_avg, map_layer3, map_fc = {}, {}, {}
    for i in tqdm(range(len(dataset))):
        img, _ = dataset[i]
        [avgpool, layer3, fc] = resnet_50_init(img, new_size, True)
        map_avg[i] = avgpool
        map_layer3[i] = layer3
        map_fc[i] = fc
    
    torch.save(map_avg, pkl_avgpool)
    torch.save(map_layer3, pkl_layer3)
    torch.save(map_fc, pkl_fc)


def compute_all_feature_extract_pickle(dataset: Caltech101):
    print("Are you sure, you want to perform this??\
          \nThis will overwrite your old pkl files, if they exist\
          \nPress Y to continue, this may take a long time to process")
    acceptance = input()
    if(acceptance == 'Y'):
        # TODO: move old pkl files to a backup folder
        # bulk_resnet(dataset=dataset, new_size=(224, 224))
        # bulk_color_moments(dataset, new_size=(300, 100))
        bulk_hog(dataset, (300, 100))