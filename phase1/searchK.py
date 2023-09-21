import torch
from tqdm import tqdm
import numpy as np
import heapq
from torchvision.datasets import Caltech101

from distances import cosine_similarity, mahalanobis, euclidean_distance, mahalanobis_with_identity, cross_correlation_distance
from utils import display_k_images_subplots

def resnet_fc_load(imageIdSearch: int, max_len: int, k: int):
    print("\n\n", "-"*15, "Resnet FC", "-"*15)
    resnet_fc = torch.load('resnet_fc.pkl')
    distances = []
    for i in tqdm(range(max_len)):
        if(i != imageIdSearch):
            distances.append(cosine_similarity( resnet_fc[imageIdSearch].flatten(), resnet_fc[i].flatten() ))
        else: distances.append(-np.inf)
    top_k = heapq.nlargest(k, enumerate(distances), key=lambda x: x[1])
    print(top_k)
    return top_k

def resnet_avgpool_load(imageIdSearch: int, max_len: int, k: int) -> tuple:
    print("\n\n", "-"*15, "Resnet Avgpool", "-"*15)
    resnet_avgpool = torch.load('resnet_avgpool.pkl')
    distances = []
    for i in tqdm(range(max_len)):
        if(i != imageIdSearch):
            distances.append(cosine_similarity( resnet_avgpool[imageIdSearch].flatten(), resnet_avgpool[i].flatten() ))
        else: distances.append(-np.inf)
    top_k = heapq.nlargest(k, enumerate(distances), key=lambda x: x[1])
    print(top_k)
    return top_k

def resnet_layer3_load(imageIdSearch: int, max_len: int, k: int):
    print("\n\n", "-"*15, "Resnet Layer3", "-"*15)
    resnet_layer3 = torch.load('resnet_layer3.pkl')
    distances = []
    for i in tqdm(range(max_len)):
        if(i != imageIdSearch):
            distances.append(cosine_similarity( resnet_layer3[imageIdSearch].flatten(), resnet_layer3[i].flatten() ))
        else: distances.append(-np.inf)
    top_k = heapq.nlargest(k, enumerate(distances), key=lambda x: x[1])
    print(top_k)
    return top_k

def color_moments_load(imageIdSearch: int, max_len: int, k: int):
    print("\n\n", "-"*15, "Color Moments", "-"*15)
    color_moments = torch.load('color_moments.pkl')
    print(len(color_moments), type(color_moments[0]), len(color_moments[0]))
    distances = []
    for i in tqdm(range(max_len)):
        if(i != imageIdSearch):
            # distances.append(mahalanobis(np.array(color_moments[imageIdSearch]), np.array(color_moments[i]) ))
            distances.append(euclidean_distance(np.array(color_moments[imageIdSearch]).flatten(), np.array(color_moments[i]).flatten() ))
            # distances.append(cosine_similarity(np.array(color_moments[imageIdSearch]).flatten(), np.array(color_moments[i]).flatten()))
        else: distances.append(np.inf)
    top_k = heapq.nsmallest(k, enumerate(distances), key=lambda x: x[1])
    print(top_k)
    return top_k

def hog_load(imageIdSearch: int, max_len: int, k: int):
    print("\n\n", "-"*15, "HOG", "-"*15)
    hog = torch.load('hog.pkl')
    # print(type(hog), len(hog), type(hog[0]))
    distances = []
    for i in tqdm(range(max_len)):
        if(i != imageIdSearch):
            # compute search
            # distances.append(mahalanobis_with_identity(hog[imageIdSearch].flatten(), hog[i].flatten()))
            # distances.append(mahalanobis(hog[imageIdSearch], hog[i]))
            distances.append(euclidean_distance(hog[imageIdSearch].flatten(), hog[i].flatten()))
        else: distances.append(np.inf)
    top_k = heapq.nsmallest(k, enumerate(distances), key=lambda x: x[1])
    print(top_k)
    return top_k

def k_resnet_layer3(imageIdSearch: int, max_val: int, k: int, dataset: Caltech101):
    top_k = resnet_layer3_load(imageIdSearch, max_val, k)
    display_k_images_subplots(dataset, top_k, 'Resnet Layer3')

def k_resnet_avgpool(imageIdSearch: int, max_val: int, k: int, dataset: Caltech101):
    top_k = resnet_avgpool_load(imageIdSearch, max_val, k)
    display_k_images_subplots(dataset, top_k, 'Resnet Avgpool')

def k_resnet_fc(imageIdSearch: int, max_val: int, k: int, dataset: Caltech101):
    top_k = resnet_fc_load(imageIdSearch, max_val, k)
    display_k_images_subplots(dataset, top_k, 'Resnet FC')

def k_color_moments(imageIdSearch: int, max_val: int, k: int, dataset: Caltech101):
    top_k = color_moments_load(imageIdSearch, max_val, k)
    display_k_images_subplots(dataset, top_k, 'Color Moments')

def k_hog(imageIdSearch: int, max_val: int, k: int, dataset: Caltech101):
    top_k = hog_load(imageIdSearch, max_val, k)
    display_k_images_subplots(dataset, top_k, 'Histograms of oriented gradients')

def process_work(imageIdSearch: int, max_val: int, k: int, dataset: Caltech101):
    try:
        k_resnet_fc(imageIdSearch, max_val, k, dataset)
        k_resnet_layer3(imageIdSearch, max_val, k, dataset)
        k_resnet_avgpool(imageIdSearch, max_val, k, dataset)
        k_color_moments(imageIdSearch, max_val, k, dataset)
        k_hog(imageIdSearch, max_val, k, dataset)
    except FileNotFoundError:
        print("Please try and run option 2 and grab some coffee!")
    # resnet_fc_load()

def compute_searchK(dataset):
    print("\n\n", "Please provide Image Id to search in all files:")
    imageIdSearch = int(input())
    print("\n\nPlease provide value for k:")
    k = int(input())
    max_val = len(dataset)
    if(imageIdSearch >= 0 & imageIdSearch < max_val):
        # display image,
        img, _ = dataset[imageIdSearch]
        img.show()
        process_work(imageIdSearch, max_val, k, dataset)
    else: print(f"Please provide valid Image Id\
            \nThe valid range is [0,{max_val}]")