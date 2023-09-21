from PIL import Image
from utils import display_image, convert_pil_tensor

from torchvision import models
import torch
import numpy as np

# def resnet_50_avgpool(img_tensor, resnet):
#     feature_vector = []

#     def hook_handler(module, input, output):
#         feature_vector.extend(output.squeeze().cpu().detach().numpy())

#     # Register the hook to the "avgpool" layer
#     hook = resnet.avgpool.register_forward_hook(hook_handler)

#     resnet.eval()
#     with torch.no_grad():
#         # to remove batch issue
#         # resnet always assumes we have a batch operating in sequence
#         resnet(img_tensor.unsqueeze(0))

#     hook.remove()
#     feature_vector = np.array(feature_vector)
#     # print(feature_vector.shape)

#     # average of consecutive entries
#     feature_vector = np.mean(feature_vector.reshape(-1,2), axis=1)
#     print(f"\n\nThere are {feature_vector.shape} entries available here...")
#     print("*"*25, 'PRINTING WHOLE ARRAY', "*"*25)
#     with np.printoptions(threshold=np.inf):
#         print(feature_vector)
#     print("*"*25, 'END OF ARRAY', "*"*25)

# def resnet_50_layer3(img_tensor, resnet):
#     feature_vector = []

#     def hook_handler(module, input, output):
#         feature_vector.extend(output.squeeze().cpu().detach().numpy())

#     # Register the hook to the "avgpool" layer
#     hook = resnet.layer3.register_forward_hook(hook_handler)

#     resnet.eval()
#     with torch.no_grad():
#         # to remove batch issue
#         # resnet always assumes we have a batch operating in sequence
#         resnet(img_tensor.unsqueeze(0))

#     hook.remove()
#     feature_vector = np.array(feature_vector)
#     # print(feature_vector.shape)

#     # average of splice of 14x14
#     feature_vector = np.mean(feature_vector, axis=(1,2))
#     print(f"\n\nThere are {feature_vector.shape} entries available here...")
#     print("*"*25, 'PRINTING WHOLE ARRAY', "*"*25)
#     with np.printoptions(threshold=np.inf):
#         print(feature_vector)
#     print("*"*25, 'END OF ARRAY', "*"*25)

# def resnet_50_fc(img_tensor, resnet):
#     feature_vector = []

#     def hook_handler(module, input, output):
#         feature_vector.extend(output.squeeze().cpu().detach().numpy())

#     # Register the hook to the "avgpool" layer
#     hook = resnet.fc.register_forward_hook(hook_handler)

#     resnet.eval()
#     with torch.no_grad():
#         # to remove batch issue
#         # resnet always assumes we have a batch operating in sequence
#         resnet(img_tensor.unsqueeze(0))

#     hook.remove()
#     feature_vector = np.array(feature_vector)
#     # print(feature_vector.shape)

#     print(f"\n\nThere are {feature_vector.shape} entries available here...")
#     print("*"*25, 'PRINTING WHOLE ARRAY', "*"*25)
#     with np.printoptions(threshold=np.inf):
#         print(feature_vector)
#     print("*"*25, 'END OF ARRAY', "*"*25)

def print_feature_vector(feature_vector):
    print(f"\n\nThere are {feature_vector.shape} entries available here...")
    print("*"*25, 'PRINTING WHOLE ARRAY', "*"*25)
    with np.printoptions(threshold=np.inf):
        print(feature_vector)
    print("*"*25, 'END OF ARRAY', "*"*25)

def fc_handle_output(fc, in_bulk: bool = False):
    feature_vector = np.array(fc)
    if(in_bulk): return feature_vector
    else: print_feature_vector(feature_vector)

def layer3_handle_output(layer3, in_bulk: bool = False):
    feature_vector = np.array(layer3)
    # average of splice of 14x14
    feature_vector = np.mean(feature_vector, axis=(1,2))

    if(in_bulk): return feature_vector
    else: print_feature_vector(feature_vector)

def avgpool_handle_output(avgpool, in_bulk: bool = False):
    feature_vector = np.array(avgpool)
    # print_feature_vector(feature_vector)
    feature_vector = np.mean(feature_vector.reshape(-1,2), axis=1)

    if(in_bulk): return feature_vector
    else: print_feature_vector(feature_vector)

def handle_input_resnet_individual(avgpool, layer3, fc):
    print("Enter corresponding input to run some task:\
        \n\n\
        1 -> Avgpool-1024\
        \n\
        2 -> Layer3-1024\
        \n\
        3 -> FC-1000\
        \n\
        0 -> quit\
        \n\n")
    option = int(input())
    match option:
        case 0: print("Exiting....")
        case 1: avgpool_handle_output(avgpool)
        case 2: layer3_handle_output(layer3)
        case 3: fc_handle_output(fc)
        case default:  print("Wrong input provided")


def resnet_50_init(pil_img: Image, new_size: tuple, in_bulk: bool = False) -> None:
    # print image
    pil_img = display_image(pil_img=pil_img, new_size=new_size, in_bulk=in_bulk)
    avgpool, layer3, fc = [], [], []

    # convert PIL image to tensor
    img_tensor = convert_pil_tensor(pil_img=pil_img)
    # take resnet model with default
    resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')

    def hook_handler_avgpool(module, input, output):
        avgpool.extend(output.squeeze().cpu().detach().numpy())

    def hook_handler_layer3(module, input, output):
        layer3.extend(output.squeeze().cpu().detach().numpy())

    def hook_handler_fc(module, input, output):
        fc.extend(output.squeeze().cpu().detach().numpy())

    # Register the hook to the "avgpool" layer
    hook_avgpool = resnet.avgpool.register_forward_hook(hook_handler_avgpool)
    hook_layer3 = resnet.layer3.register_forward_hook(hook_handler_layer3)
    hook_fc = resnet.fc.register_forward_hook(hook_handler_fc)

    resnet.eval()
    with torch.no_grad():
        # to remove batch issue
        # resnet always assumes we have a batch operating in sequence
        resnet(img_tensor.unsqueeze(0))

    hook_avgpool.remove()
    hook_layer3.remove()
    hook_fc.remove()

    # refactor this to become np array
    # feature_vector = np.array(feature_vector)

    # ask user for input so as to select correct layer
    if(in_bulk): return [avgpool_handle_output(avgpool, in_bulk), layer3_handle_output(layer3, in_bulk), fc_handle_output(fc, in_bulk)]
    else: handle_input_resnet_individual(avgpool, layer3, fc)