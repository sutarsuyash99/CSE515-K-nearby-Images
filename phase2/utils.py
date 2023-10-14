from typing import Tuple
import math
from collections import defaultdict
from PIL import ImageOps
import PIL
import torchvision
import torch
from matplotlib import pyplot
from torchvision import datasets
from PIL import Image
import numpy as np
import pandas as pd
import os
import distances
# from ordered_set import OrderedSet

pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 30)
pd.set_option('display.min_rows', 20)
pd.set_option('display.precision', 2)


from Mongo.mongo_query_np import get_all_feature_descriptor

distance_function_per_feature_model = {
    1: distances.euclidean_distance,
    2: distances.cosine_distance,
    3: distances.cosine_distance,
    4: distances.cosine_distance,
    5: distances.cosine_distance,
    6: distances.kl_divergence
}

feature_model  = {
    1 : "color_moment",
    2 : "hog",
    3 : "avgpool",
    4 : "layer3",
    5 : "fc_layer",
    6 : "resnet_final"
}

latent_semantics = {
    1 : "SVD",
    2 : "NNMF",
    3 : "LDA",
    4 : "K_means"
}

def select_distance_function_for_model_space(option: int):
    if option in distance_function_per_feature_model:
        return distance_function_per_feature_model[option]
    return None

def int_input(default_value: int = 99) -> int:
    try:
        inpu = int(input())
        return inpu
    except ValueError:
        print(f'No proper value was passed, Default value of {default_value} was used')
        return default_value

def convert_higher_dims_to_2d(data_collection: np.ndarray) -> np.ndarray:
    '''
    Converts higher dimension vector to 2d vector
    Parameters:
        data_collection: numpy.ndarray vector of higher dimensions
    Returns:
        returns numpy.ndarray vector of two dimensions
    '''
    if data_collection.ndim > 2:
        og_shape = data_collection.shape
        new_shape = (og_shape[0], np.prod(og_shape[1:]))
        data_collection = data_collection.reshape(new_shape)
    return data_collection

# for query image id, return label name for it
def name_for_label_index(dataset: torchvision.datasets.Caltech101, index: int) -> str:
    dataset_named_categories = dataset.categories
    return dataset_named_categories[index]

# for query image id, return (PIL image, label_id, label_name) 
# returns IndexError if index is not in range
def img_label_and_named_label_for_query_int(dataset: torchvision.datasets.Caltech101, 
                                            index: int) -> Tuple[any, int, str]:
    if(index > len(dataset) or index < 0): 
        print("Not proper images")
        return IndexError
    else:
        img, label_id = dataset[index]
        label_name = name_for_label_index(dataset, label_id)
        return (img, label_id, label_name)

def initialise_project():
    dataset = torchvision.datasets.Caltech101(root='./data', download=True, target_type='category')

    # this is going to be created once and passed throughout in all functions needed
    # dict: (label: string, list<imageIds: int>)
    # where key is either label index as int (eg: faces is with index 0) or it is the label name as string
    # both are added to the map, user can decide which to use
    # value is the list of ids belonging to that category
    labelled_images = defaultdict(list)
    dataset_named_categories = dataset.categories
    for i in range(len(dataset)):
        img, label = dataset[i]
        # label returns the array index for dataset.categories
        # labelled_images[str(label)].append(i)
        category_name = dataset_named_categories[label]
        labelled_images[category_name].append(i)
    return (dataset, labelled_images)

def get_image_categories():
    dataset = torchvision.datasets.Caltech101(root='./data', download=True, target_type='category')
    labelled_images = defaultdict(list)
    dataset_named_categories = dataset.categories 
    for i in range(len(dataset)):
        _, label = dataset[i]
        category_name = dataset_named_categories[label]
        labelled_images[i] = category_name
    return labelled_images

def display_image_og(pil_img)->Image:
    pil_img.show()
    return pil_img

def find_nearest_square(k: int) -> int:
    return math.ceil(math.sqrt(k))

def gen_unique_number_from_title(string: str) -> int:
    a = 0
    for c in string:
        a+=ord(c)
    return a

def display_k_images_subplots(dataset: datasets.Caltech101, distances: tuple, title: str):
    pyplot.close()
    k = len(distances)
    # print(len(distances))
    # distances tuple 0 -> id, 1 -> distance
    split_x = find_nearest_square(k)
    split_y = math.ceil(k/split_x)
    # print(split_x, split_y)
    # this does not work
    # pyplot.figure(gen_unique_number_from_title(title))
    fig, axs = pyplot.subplots(split_x, split_y)
    fig.suptitle(title)
    ii = 0
    for i in range(split_x):
        for j in range(split_y):
            if(ii < k):
                # print(ii)
                id, distance = distances[ii][0], distances[ii][1]
                img, _ = dataset[id]
                if(img.mode == 'L'): axs[i,j].imshow(img, cmap = 'gray')
                else: axs[i,j].imshow(img)
                axs[i,j].set_title(f"Image Id: {id} Distance: {distance:.2f}")
                axs[i,j].axis('off')
                ii += 1
            else:
                fig.delaxes(axs[i,j])
    # pyplot.title(title)
    pyplot.show()

def get_user_selected_feature_model(): 
    """This is a helper code which prints all the available fearure options and takes input from the user"""
    print('Select your option:\
        \n\n\
        \n1. Color Moments\
        \n2. Histogram of Oriented gradients\
        \n3. RESNET-50 Avgpool\
        \n4. RESNET-50 Layer3\
        \n5. RESNET-50 FC\
        \n6. RESNET Final\
        \n\n')
    option = int_input()
    model_space = None
    dbName = None
    match option:
        case 1: dbName = 'color_moment'
        case 2: dbName = 'hog'
        case 3: dbName = 'avgpool'
        case 4: dbName = 'layer3'
        case 5: dbName = 'fc_layer'
        case 6: dbName = 'resnet_final'
        case default: print('No matching input was selected')
    if dbName is not None:
        model_space = get_all_feature_descriptor(dbName)
    return model_space, option, dbName

def get_user_selected_feature_model_only_resnet50_output():
    """This is a helper code which prints resnet50_final layer output"""
    print("Model space in use -----> RESNET-50 Final layer (softmax) values")
    dbName = 'resnet_final'
    model_space = get_all_feature_descriptor(dbName)
    # returning 2nd parameter to make function syntatically similar to get_user_selected_feature_model function
    return model_space, None, dbName

def get_user_input_image_id():
    """Helper function to be used in other function to take image Id from user"""
    print("Please enter the value of ImageId: ")
    return int_input()

def get_user_input_label():
    """Helper function to be used in other function to take label"""
    print("Please enter the value of label: ")
    label = int_input(0)
    return label

def get_user_input_for_saved_files(option: int):
    """
    Helper function which prints all available files and return file name relative from source file
    Returns:
        pathname: relative path to pkl from phase root folder (None if no pkl exists)
    """
    base_path = '/LatentSemantics/LS'+str(option)
    dir = os.getcwd() + base_path
    onlyfiles = []
    for f in os.listdir(dir):
        if (os.path.isfile(os.path.join(dir, f)) and f != '.gitkeep'):
            onlyfiles.append(f)
    if len(onlyfiles) == 0:
        print('No models saved -- please run task 3-6 accordingly')
        return None
    else:
        print("Please select the option file name you want")
        for i in range(len(onlyfiles)):
            print(f'{i} -> {onlyfiles[i]}')
        print('\n')
        index = int_input(0)
        return '.' + base_path + '/' + onlyfiles[index]

def get_user_input_latent_semantics():
    """
    Helper function to be used in other functions to take user input
    Parameters: None
    Returns: 
        path: Path to model file
        LS-option: option need
    LS1, LS2, LS3, LS4
    LS1: SVD, NNMF, k-means, LDA
    LS2: CP-decompositions
    LS3: Label-Label similarity -> reduced space
    LS4: Image-Image similarity -> reduced space
    """
    print('\n\nSelect your Latent Space: \
          \n1. LS1 --> SVD, NNMF, kMeans, LDA\
          \n2. LS2 --> CP-decomposition\
          \n3. LS3 --> Label-Label similarity\
          \n4. LS4 --> Image-Image similarity\n')
    option = int_input(1)
    path_to_model = None
    match option:
        case 1: path_to_model = get_user_input_for_saved_files(1)
        case 2: path_to_model = get_user_input_for_saved_files(2)
        case 3: path_to_model = get_user_input_for_saved_files(3)
        case 4: path_to_model = get_user_input_for_saved_files(4)
        case default: print('Nothing here ---> wrong input provided')
    return path_to_model, option

def get_user_external_image_path():
    """
    Helper function to be used in other functions to get external image path 
    and open and show image
    """
    print("Please provide path to image as input: consider full path")
    file_path = input()
    try:
        img = Image.open(file_path)
        return img
    except FileNotFoundError:
        print(f"File path not improper, file does not exist")
    except PIL.UnidentifiedImageError:
        print("Image could not be indentified or opened")
    except Exception as e:
        print(f'There was {e} encountered')
    return None

def get_user_input_internalexternal_image():
    """
    Helper function to be used to either get an external/internal image
    parameters:
        None
    Returns:
        Tuple: (ImageId, PIL)
            ImageId: returns ImageId if internal dataset, if external -1 will be returned
            PIL: None if internal dataset, for external file it will be non None
    internal image: present in dataset, may or may not be present in database
    external image: not in dataset and not in database
    """
    print("Select your option:\
          \n\n\
          \n1. Dataset Image\
          \n2. External Image\
          \n\n")
    selection = int_input(1)
    if selection == 1:
        return (get_user_input_image_id(), None) 
    else:
        # take path from user
        return (-1, get_user_external_image_path())


def get_user_input_k():
    """Helper function to be used in other functions to take input from user for K"""
    print("Please enter the value of K : ")
    value = int_input()
    return value

def get_user_selected_dim_reduction():
    """This is a helper code which prints all the available Dimension reduction options and takes input from the user"""
    print('Select your option:\
        \n\n\
        \n1. SVD - Singular Value Decomposition\
        \n2. NNMF - Non Negative Matrix Factorization\
        \n3. LDA - Latent Dirichlet Allocation\
        \n4. K - Means\
        \n\n')
    option = int_input()
    return option

def print_decreasing_weights(data, object = "ImageID"):
    dataset = torchvision.datasets.Caltech101(root='./data', download=True, target_type='category')
    m, n = data.shape
    df = pd.DataFrame()
    for val in range(n):
        ls = data[: ,val]
        # ls = np.round(ls, 2)
        indexed_list = list(enumerate(ls))
        sorted_list = sorted(indexed_list, key=lambda x: x[1])
        sorted_list.reverse()
        if object == "ImageID":
            sorted_list = [(x * 2, y) for x, y in sorted_list]
        else :
            sorted_list = [(name_for_label_index(dataset, x) , y) for x, y in sorted_list]
                
        df["LS"+str(val+1) + "  "+ object + ", Weights"] = sorted_list
        df.index.name = "Rank"
    print("Output Format - ImageID, Weight")
    print(df)

def get_cv2_image(image_id):
    """Given an Image ID this function return the image in cv2/numpy format"""
    dataset = datasets.Caltech101(root="./", download=True)
    image, label = dataset[int(image_id)]
    # Converting the Image to opencv/numpy format
    cv2_image = np.array(image)
    return cv2_image

def get_cv2_image_grayscale(image_id):
    """Given an Image ID this function returns the grayscale image in cv2/numpy format"""
    dataset = datasets.Caltech101(root="./", download=True)
    image, label = dataset[int(image_id)]
    # Converting the Image to opencv/numpy format
    pil_img = ImageOps.grayscale(image)
    # cv2_image = np.array(image)
    cv2_image = np.array(pil_img)
    return cv2_image

def check_rgb_change_grayscale_to_rgb(image):
    """This function check if the image is rgb and if not then converts the grayscale image to rgb"""

    if image.mode == 'RGB':
        return image
    else:
        rgb_image = image.convert('RGB')
        return rgb_image
    
def convert_image_to_grayscale(image):
    """Converts the pil image to grayscale and then converting to cv2"""

    gray_image = ImageOps.grayscale(image)
    cv2_image = np.array(gray_image)

    return cv2_image

def compute_distance_query_image_top_k(
        k: int,
        labelled_feature_vectors: dict,
        model_space: np.ndarray,
        cur_label: str,
        option: int,
    ) -> list:
        """
        Function that computes top k images for label in given model space
        Parameters:
            k: integer
            labelled_feature_vectors: (dict) (key: label name) -> value: label_feature vector
            model_space: (list) which contains feature vectors for all images
            cur_label: (str) label index name -- eg. 'Faces'
            option: (int) internal map index -- see: feature_model
        """
        top_distances = []
        cur_label_fv = labelled_feature_vectors[cur_label]

        distance_function_to_use = select_distance_function_for_model_space(option)

        for i in range(len(model_space)):
            distance = distance_function_to_use(
                cur_label_fv.flatten(),
                model_space[i].flatten(),
            )
            top_distances.append((distance, i * 2))
        top_distances.sort()

        top_k = []
        for i in range(k):
            top_k.append((top_distances[i][1], top_distances[i][0]))

        print("-" * 20)
        for i in top_k:
            print(f"ImageId: {i[0]}, Distance: {i[1]}")
        print("-" * 20)
        # list of tuple
        # -------------
        # 0: id
        # 1: distance
        return top_k