from typing import Tuple
import math
from collections import defaultdict
from PIL import ImageOps, Image
import PIL
import torchvision
import torch
from matplotlib import pyplot
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import distances
import glob
import distances
from tqdm import tqdm
import heapq

from resnet_50 import resnet_features
import Mongo.mongo_query_np as mongo_query
from sklearn.preprocessing import StandardScaler,normalize,MinMaxScaler
# from ordered_set import OrderedSet

pd.set_option("display.max_rows", 30)
pd.set_option("display.max_columns", 30)
pd.set_option("display.min_rows", 20)
pd.set_option("display.precision", 2)


from Mongo.mongo_query_np import (
    get_all_feature_descriptor,
    get_label_feature_descriptor,
)

distance_function_per_feature_model = {
    1: distances.euclidean_distance,
    2: distances.cosine_distance,
    3: distances.cosine_distance,
    4: distances.cosine_distance,
    5: distances.cosine_distance,
    6: distances.kl_divergence,
}

feature_model = {
    1: "color_moment",
    2: "hog",
    3: "avgpool",
    4: "layer3",
    5: "fc_layer",
    6: "resnet_final",
}

label_feature_model = {5: "label_fc_vectors"}

latent_semantics = {1: "SVD", 2: "NNMF", 3: "LDA", 4: "K_means"}


def select_distance_function_for_model_space(option: int):
    """
    Takes the feature space as input gives function which is best suited to calculate the distance
    Input  - feature_model key (see above)
    Output - Suitable distance function for the given feature space/model
    """
    if option in distance_function_per_feature_model:
        return distance_function_per_feature_model[option]
    return None


def int_input(default_value: int = 99) -> int:
    try:
        inpu = int(input())
        return inpu
    except ValueError:
        print(f"No proper value was passed, Default value of {default_value} was used")
        return default_value


def convert_higher_dims_to_2d(data_collection: np.ndarray) -> np.ndarray:
    """
    Converts higher dimension vector to 2d vector
    Parameters:
        data_collection: numpy.ndarray vector of higher dimensions
    Returns:
        returns numpy.ndarray vector of two dimensions
    """
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
def img_label_and_named_label_for_query_int(
    dataset: torchvision.datasets.Caltech101, index: int
) -> Tuple[any, int, str]:
    if index > len(dataset) or index < 0:
        print("Not proper images")
        return IndexError
    else:
        img, label_id = dataset[index]
        label_name = name_for_label_index(dataset, label_id)
        return (img, label_id, label_name)


def get_labels():
    dataset = torchvision.datasets.Caltech101(
        root="./data", download=True, target_type="category"
    )
    dataset_named_categories = dataset.categories
    return dataset_named_categories


def initialise_project():
    """
    Creates dataset and then creates labblled_images dictionary containing(label: string, list<imageIds: int>)
    Input : None
    Output : dataset, labelled_images(dict - (label: string, list<imageIds: int>))
    """
    dataset = torchvision.datasets.Caltech101(
        root="./data", download=True, target_type="category"
    )

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
    dataset = torchvision.datasets.Caltech101(
        root="./data", download=True, target_type="category"
    )
    labelled_images = defaultdict(list)
    dataset_named_categories = dataset.categories
    for i in range(len(dataset)):
        _, label = dataset[i]
        category_name = dataset_named_categories[label]
        labelled_images[i] = category_name
    return labelled_images


def display_image_og(pil_img) -> Image:
    pil_img.show()
    return pil_img


def find_nearest_square(k: int) -> int:
    return math.ceil(math.sqrt(k))


def gen_unique_number_from_title(string: str) -> int:
    a = 0
    for c in string:
        a += ord(c)
    return a


def display_k_images_subplots(
    dataset: datasets.Caltech101, distances: tuple, title: str, pil_image=None
):
    """
    This function display the images which is passed in disantances tuple and external or odd image given in pil_image
    Input : dataset , distance - tuple(imageid, distance), title - plot title, pil_image = Odd or External image
    Output : None (Diplays images using pyplot)
    """
    pyplot.close()
    k = len(distances)
    if pil_image is not None:
        pil_k = k + 1
    else:
        pil_k = k
    # print(len(distances))
    # distances tuple 0 -> id, 1 -> distance
    split_x = find_nearest_square(pil_k)
    split_y = math.ceil(pil_k / split_x)

    # print(split_x, split_y)
    # this does not work
    # pyplot.figure(gen_unique_number_from_title(title))
    fig, axs = pyplot.subplots(split_x, split_y)
    fig.suptitle(title)
    ii = 0
    for i in range(split_x):
        for j in range(split_y):
            if ii < k:
                # print(ii)
                if i == 0 and j == 0 and pil_image is not None:
                    img = pil_image
                    if img.mode == "L":
                        axs[i, j].imshow(img, cmap="gray")
                    else:
                        axs[i, j].imshow(img)
                    axs[i, j].set_title(f"Query Image", fontsize=12)
                    axs[i, j].axis("off")
                    # ii += 1
                else:
                    id, distance = distances[ii][0], distances[ii][1]
                    img, _ = dataset[id]
                    if img.mode == "L":
                        axs[i, j].imshow(img, cmap="gray")
                    else:
                        axs[i, j].imshow(img)
                    axs[i, j].set_title(
                        f"Image Id: {id} Distance: {distance:.2f}", fontsize=10
                    )
                    axs[i, j].axis("off")
                    ii += 1
            else:
                fig.delaxes(axs[i, j])
    # pyplot.title(title)
    pyplot.show()


def get_user_selected_feature_model():
    """This is a helper code which prints all the available fearure options and takes input from the user"""
    print(
        "Select your option:\
        \n\n\
        \n1. Color Moments\
        \n2. Histogram of Oriented gradients\
        \n3. RESNET-50 Avgpool\
        \n4. RESNET-50 Layer3\
        \n5. RESNET-50 FC\
        \n6. RESNET Final\
        \n\n"
    )
    option = int_input()
    model_space = None
    dbName = None
    match option:
        case 1:
            dbName = "color_moment"
        case 2:
            dbName = "hog"
        case 3:
            dbName = "avgpool"
        case 4:
            dbName = "layer3"
        case 5:
            dbName = "fc_layer"
        case 6:
            dbName = "resnet_final"
        case default:
            print("No matching input was selected")
    if dbName is not None:
        model_space = get_all_feature_descriptor(dbName)
    return model_space, option, dbName


def get_user_selected_feature_model_only_resnet50_output():
    """This is a helper code which prints resnet50_final layer output"""
    print("Model space in use -----> RESNET-50 Final layer (softmax) values")
    dbName = "resnet_final"
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

def get_user_input_odd_image_id_looped(dataset : torchvision.datasets.Caltech101 ) -> int :

    '''
    Helper function to get odd image id from the user.
    Loops until it receives 'x' to exit or odd image id present in dataset.
    Returns either 'x'  or present in dataset.
    '''
    while True :

        user_input = input(f"\nEnter 'x' to exit or odd image id to get predicted label : ")
        if user_input.lower() == 'x' :
            return 'x'
        
        if user_input.isdigit():
            user_input = int(user_input)
            if user_input % 2 == 0:
                print("Enter a valid odd image id.")
            elif user_input > len(dataset):
                print("Image id not in the dataset. Enter a valid odd image id.")
            else:
                return user_input
        else :
            print(f"Enter a valid odd image id.") 



def get_user_selection_classifier():
    """This is a helper code which prints all the available classifiers code and take input from user"""
    print(
        "Select your option:\
          \n\n\
          \n1. m-NN\
          \n2. Decision Tree\
          \n3. PPR classifier\
          \n\n"
    )
    option = int_input(3)
    return option


def get_user_input_for_saved_files(option: int):
    """
    Helper function which prints all available files and return file name relative from source file
    Returns:
        pathname: relative path to pkl from phase root folder (None if no pkl exists)
    """
    base_path = "/LatentSemantics/LS" + str(option)
    dir = os.getcwd() + base_path
    onlyfiles = []
    for f in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, f)) and f != ".gitkeep":
            onlyfiles.append(f)
    if len(onlyfiles) == 0:
        print("No models saved -- please run task 3-6 accordingly")
        return None
    else:
        print("Please select the option file name you want")
        for i in range(len(onlyfiles)):
            print(f"{i} -> {onlyfiles[i]}")
        print("\n")
        index = int_input(0)
        return "." + base_path + "/" + onlyfiles[index]


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
    print(
        "\n\nSelect your Latent Space: \
          \n1. LS1 --> SVD, NNMF, kMeans, LDA\
          \n2. LS2 --> CP-decomposition\
          \n3. LS3 --> Label-Label similarity\
          \n4. LS4 --> Image-Image similarity\n"
    )
    option = int_input(1)
    path_to_model = None
    match option:
        case 1:
            path_to_model = get_user_input_for_saved_files(1)
        case 2:
            path_to_model = get_user_input_for_saved_files(2)
        case 3:
            path_to_model = get_user_input_for_saved_files(3)
        case 4:
            path_to_model = get_user_input_for_saved_files(4)
        case default:
            print("Nothing here ---> wrong input provided")
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
        print(f"There was {e} encountered")
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
    print(
        "Select your option:\
          \n\n\
          \n1. Dataset Image\
          \n2. External Image\
          \n\n"
    )
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
    print(
        "Select your option:\
        \n\n\
        \n1. SVD - Singular Value Decomposition\
        \n2. NNMF - Non Negative Matrix Factorization\
        \n3. LDA - Latent Dirichlet Allocation\
        \n4. K - Means\
        \n\n"
    )
    option = int_input()
    return option


def print_decreasing_weights(data, object="ImageID"):
    """
    Converts Nd array into pandas dataframe and prints it in decreasing order
    Input - data : Label/weight pairs
            object : either ImageID or Label
    Output - None (Prints data in decreasing order)
    """
    dataset = torchvision.datasets.Caltech101(
        root="./data", download=True, target_type="category"
    )
    m, n = data.shape
    df = pd.DataFrame()
    for val in range(n):
        ls = data[:, val]
        # ls = np.round(ls, 2)
        indexed_list = list(enumerate(ls))
        sorted_list = sorted(indexed_list, key=lambda x: x[1])
        sorted_list.reverse()
        if object == "ImageID":
            sorted_list = [(x * 2, y) for x, y in sorted_list]
        else:
            sorted_list = [
                (name_for_label_index(dataset, x), y) for x, y in sorted_list
            ]

        df["LS" + str(val + 1) + "  " + object + ", Weights"] = sorted_list
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

    if image.mode == "RGB":
        return image
    else:
        rgb_image = image.convert("RGB")
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


def get_user_input_model_or_space():
    """Helper function to be used in other function to take model or space"""
    print(
        "\n\nSelect : \
          \n1. Feature model\
          \n2. Latent space and feature model\n"
    )

    option = int_input()
    return option


def get_saved_model_files(
    feature_model: str, latent_space: int = None, d_reduction: str = None
):
    """
    Helper function which check for model files.
    Returns:
        pathname: relative path to pkl from phase root folder (None if no pkl exists)
    """

    # Case 1 : Only feature_model no latent_semantics
    if latent_space == None:
        pattern = f"image_image_similarity_matrix_{feature_model}*.pkl"
    # Case 2 : CP-decomposition
    elif latent_space == 2:
        pattern = f"LS{latent_space}_{feature_model}*.pkl"
    # Case 3 : Latent semantics and feature_model
    elif latent_space != None:
        pattern = f"LS{latent_space}_{feature_model}_{d_reduction}*.pkl"

    current_directory = os.getcwd()

    # Initialize a list to store matching file paths
    matching_files = []

    # Recursively search for files in successive directories
    for root, dirs, files in os.walk(current_directory):
        for file_path in glob.glob(os.path.join(root, pattern)):
            matching_files.append(file_path)

    if len(matching_files) > 0:
        matching_file = matching_files[0]
        # print(matching_file)
        return matching_file
    else:
        return None


def get_user_selected_latent_space_feature_model():
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
    print(
        "\n\nSelect your Latent Space: \
          \n1. LS1 --> SVD, NNMF, kMeans, LDA\
          \n2. LS2 --> CP-decomposition\
          \n3. LS3 --> Label-Label similarity\
          \n4. LS4 --> Image-Image similarity\n"
    )
    ls_option = int_input(1)

    # Get feature model for latent space
    _, fs_option, _ = get_user_selected_feature_model()

    # Get dimensionality reduction
    if ls_option != 2:
        dr_option = get_user_selected_dim_reduction()
        return ls_option, fs_option, dr_option
    return ls_option, fs_option, None


def get_label_vectors(ls_option: int) -> np.ndarray:
    label_model_space = label_feature_model[ls_option]
    label_space = get_label_feature_descriptor(label_model_space)
    # convert to 4339, fx
    label_space = convert_higher_dims_to_2d(label_space)
    return label_space


def get_odd_image_feature_vectors(feature_model : str ) -> np.ndarray :

    '''
    Helper function to get odd image feature vectors
    Currently supports fc layer only 
    '''
    match feature_model :

        case 'fc_layer' :
            feature_path = 'fc_layer_vectors.pkl'
    
    
    if not os.path.isfile(feature_path) :
        print(f"Generate RESNET FC features for odd images...")
        return None
    else :
        odd_images = torch.load(feature_path)
        odd_image_vectors = np.vstack([i for index,i in enumerate(odd_images.values()) if index % 2 != 0])
        
    
    return odd_image_vectors





def generate_image_similarity_matrix_from_db(
    feature_model: str, fs_option: int
) -> np.ndarray:
    data = mongo_query.get_all_feature_descriptor(feature_model)

    N = data.shape[0]
    distance_matrix = np.zeros((N, N))
    distance_function_to_use = select_distance_function_for_model_space(fs_option)

    # If feature space is color_moment or hog use cosine or use euclidean
    for i in tqdm(range(N)):
        for j in range(N):
            # Calculate the similarity using your similarity function
            distance = distance_function_to_use(data[i].flatten(), data[j].flatten())
            distance_matrix[i, j] = distance

    torch.save(
        distance_matrix,
        f"./LatentSemantics/LS4/image_image_matrix/image_image_similarity_matrix_{feature_model}.pkl",
    )
    return distance_matrix


def generate_matrix_from_image_weight_pairs(
    data: np.ndarray, fs_option: int
) -> np.ndarray:
    """
    Generates image_image similarity matrix from image-weight pairs
    """
    N = data.shape[0]
    distance_matrix = np.zeros((N, N))
    distance_function_to_use = select_distance_function_for_model_space(fs_option)
    # If feature space is color_moment or hog use cosine or use euclidean
    for i in tqdm(range(N)):
        for j in range(N):
            # Calculate the similarity using your similarity function
            distance = distance_function_to_use(data[i].flatten(), data[j].flatten())
            distance_matrix[i, j] = distance

    return distance_matrix


def get_closest_label_for_image(
    label_vectors: np.ndarray,
    image_features: np.ndarray,
    model_space_selection: int,
    top_k: int,
) -> list:
    """
    Helper function to get closest label vector to the input feature vector space
    Inputs:
        label_vectors: all the vectors of feature space, currently only fc is in DB
        image_features: feature vectors of image
        model_space_selection: int option to select distance formula to use
        top_k: value of top distances to return
    Returns:
        List of tuples of distances with index, distance to it
    """
    distance_fn_to_use = distance_function_per_feature_model[model_space_selection]
    distance_heaps = []
    for i in range(len(label_vectors)):
        cur_distance = distance_fn_to_use(
            label_vectors[i].flatten(), image_features.flatten()
        )
        heapq.heappush(distance_heaps, (cur_distance, i))

    top_k_labels = []
    for i in range(top_k):
        if len(distance_heaps) == 0: break
        cur_distance, index = heapq.heappop(distance_heaps)
        top_k_labels.append((index, cur_distance))
    return top_k_labels


def get_closest_image_from_db_for_image(
    image_id: int,
    image_features: np.ndarray,
    model_space_selection: int,
    top_k: int,
    dataset,
) -> list:
    """
    Helper function to get closest image from db from image not in DB
    Currently it runs only resnet, small change can fix that
    inputs:
        image_id: user input for image id -> check in place for IndexError: will return None
        image_features: all image features in DB
        model_space_selection: parameter to decide which feature space to use -> distances can be mapped to this
        top_k: list of top k closest values
        dataset: to get the image PIL from image_id
    Returns:
        List of tuples of distances with index, distance to it
    """
    img = -1
    try:
        img, _ = dataset[image_id]
    except IndexError:
        print('Error input was provided')
        return None

    resnet = resnet_features()
    
    resnet.run_model(img)
    image_vector = resnet.resnet_fc_layer()
    
    distance_fn_to_use = distance_function_per_feature_model[model_space_selection]
    distance_heaps = []
    
    for i in range(len(image_features)):
        cur_distance = distance_fn_to_use(
            image_vector.flatten(), image_features[i].flatten()
        )
        heapq.heappush(distance_heaps, (cur_distance, i))

    top_k_labels = []
    for i in range(top_k):
        if len(distance_heaps) == 0: break
        cur_distance, index = heapq.heappop(distance_heaps)
        top_k_labels.append((index*2, cur_distance))
    
    return top_k_labels

def get_user_input_numeric_common(default_val, variable_name):
    """Helper function to get the numeric input for variable in question"""
    print(f"Enter the value for {variable_name}:")
    return int_input(default_val)


def get_odd_image_ids(dataset) -> list :

    '''
    Returns list of only odd image ids from the dataset
    '''
    return [ i for i in range(1,len(dataset),2) ] 


def cosine_similarity_matrix( a: np.ndarray, b: np.ndarray ) -> np.ndarray :

    '''
    Returns a similarity matrix with cosine similarity scores w.r.t. each item in the given matrices.
    '''
    norm_a = a / np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return  np.dot(norm_a, norm_b.T)  


def cosine_distance_matrix( a: np.ndarray, b: np.ndarray ) -> np.ndarray :

    '''
    Returns a distance matrix with cosine distance scores w.r.t. each item in the given matrices.
    '''
    similarity_matrix = cosine_similarity_matrix(a,b)
    cosine_distance_matrix = 1 - similarity_matrix

    #Ensuring that diagonals are zero, removing precision errors
    threshold = 1e-12
    cosine_distance_matrix[cosine_distance_matrix < threshold] = 0
    
    return  cosine_distance_matrix


def euclidean_distance_matrix( a: np.ndarray, b: np.ndarray ) -> np.ndarray :

    '''
    Returns a distance matrix with euclidean distance scores w.r.t. each item in the given matrices.
    '''
   
    #Cannot be used due to space constraints in createing new dimension
    #return np.linalg.norm(a[:, np.newaxis] - b, axis=-1)
   
    squared_euclidean_matrix = -2 * np.dot(a, b.T) + np.sum(a**2, axis=1, keepdims=True) + np.sum(b**2, axis=1, keepdims=True).T
    
    #Ensuring that diagonals are zero, removing precision errors
    squared_euclidean_matrix[squared_euclidean_matrix < 0] = 0
    threshold = 1e-12
    squared_euclidean_matrix[squared_euclidean_matrix < threshold] = 0

    #Distance matrix
    euclidean_distance_matrix = np.sqrt(squared_euclidean_matrix)

    return euclidean_distance_matrix


def zscore_normalization(data : np.ndarray) -> np.ndarray :

    '''
    Returns zscore normalized values for the input matrix 
    '''

    scaler = StandardScaler()
    return scaler.fit_transform(data)


def l2_normalization(data : np.ndarray) -> np.ndarray :

    '''
    Returns l2 normalized values for the input matrix 
    '''

    return normalize(data, norm='l2')


def MinMax_normalization(data : np.ndarray) -> np.ndarray :

    '''
    Returns min max scaler normalized values for the input matrix 
    '''
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)




def compute_scores(actual : np.ndarray , predicted : np.ndarray, avg_type : str = None, values : bool = False)  :

    '''
    Creates confusion matrix based on actual [ rows ] and predicted [columns] values
    Returns either confusion matrix or scores [ PRECISION , RECALL, F1, ACCURACY ] depending upon set value
    Input arrays need to contain integers or floats.
    '''

    if len(actual) != len(predicted) :
        raise ValueError(f"Length of actual and predicted values need to be same")

    #Number of classes
    N = len(np.unique(actual)) 
    confusion_matrix = np.zeros((N, N))
    
    for i in range(len(actual)):
        #print(f"i : {i} - actual : actual[{i}] - {actual[i]}, predicted : predicted[{i}] - {predicted[i]}")
        confusion_matrix[int(actual[i])][int(predicted[i])] += 1
    
    confusion_matrix = confusion_matrix.astype(int)
    

    #Calculating TP,FP,TN,FN form confusion matrix per class 
    if values :

        #Same class values that are in diagonal, If the predicted value is positive and the actual value is positive then its true positive 
        true_positives = confusion_matrix.diagonal()

        #Sum of all values per row minus the diagonal values i.e true positives 
        false_negatives =  confusion_matrix.sum(axis=1) - true_positives
        
        #Sum of all values per columns minus the diagonal values i.e true positives, If the predicted value is positive and the actual value is negative then its false positive 
        false_positives = confusion_matrix.sum(axis=0) - true_positives


        # Sum of all values per row and column minus the diagonal values i.e true positives, If the predicted value is negative and the actual value is negative then it's true negative
        #https://stackoverflow.com/questions/31345724/scikit-learn-how-to-calculate-the-true-negative
        true_negatives =  confusion_matrix.sum() - false_negatives - false_positives - true_positives


        match avg_type :

            case  None :

                '''
                If average type is none calculate precision, f1, recall per class and return an array
                '''
                precision = true_positives/(true_positives + false_positives)
                recall    = true_positives/(true_positives + false_negatives)
                f1  = 2 * ((precision*recall) / (precision + recall))

                
            case 'micro' :
                
                '''
                If average type is micro calculate precision, f1, recall globally by adding everything
                '''
                precision = true_positives.sum()/(true_positives.sum() + false_positives.sum())
                recall    = true_positives.sum()/(true_positives.sum() + false_negatives.sum())
                f1  = 2 * ((precision*recall) / (precision + recall))


            case 'macro' :
                
                '''
                If average type is macro calculate precision, f1, recall per class and take unweighted mean
                '''
                precision = true_positives/(true_positives + false_positives)
                recall    = true_positives/(true_positives + false_negatives)
                f1  = 2 * ((precision*recall) / (precision + recall))
                
                precision = precision.sum()/len(precision)
                recall = recall.sum()/len(recall)
                f1 =  f1.sum()/len(f1)

            case 'weighted' :
                
                '''
                If average type is weighted calculate precision, f1, recall per class and take weighted mean
                '''

                
                #frequency for each label 
                _ , per_class_frequency = np.unique(actual, return_counts = True)

                #precision same as None i.e for each class 
                precision_none = true_positives/(true_positives + false_positives)
                recall_none    = true_positives/(true_positives + false_negatives)
                f1_none  = 2 * ((precision_none*recall_none) / (precision_none + recall_none))

                precision = np.sum(precision_none * (per_class_frequency / np.sum(per_class_frequency)))
                recall =  np.sum(recall_none * (per_class_frequency / np.sum(per_class_frequency)))
                f1 = np.sum(f1_none * (per_class_frequency / np.sum(per_class_frequency)))

        
        # In binomial i.e only two class accuracy is defined as (TP + TN)/(TP + TN + FP + FN)
        # But for multiclass : Number of correct predictions / Number of predictions made    
        accuracy =  confusion_matrix.diagonal().sum() / len(actual)
        return precision, recall, f1, accuracy

    #In case values is false provide the confusion matrix itself
    return confusion_matrix


def print_scores_per_label(dataset : torchvision.datasets.Caltech101, precision : np.ndarray, recall : np.ndarray, f1 : np.ndarray, accuracy : float, name : str) -> None :

    '''
    Prints the score report for every label i.e precision, recall, f1  as well as accuracy score for the classifier.
    Input : precision, recall, f1 arrays labelwise, accuracy score [ Output of compute score ] and classifier name.
    Output : None
    '''
    spacing = '\t'
    print(f"\n\n{spacing} Classification score report for model : {name} {spacing}\n")

    print(f"{spacing} Accuracy : {accuracy}  /  {accuracy*100:.2f} %  {spacing}\n")

    #Constructing DataFrame containing label id , label name, precision, recall, f1 scores 
    data = []
    for i in range(len(precision)) :
        data.append([ i , name_for_label_index(dataset, i), precision[i], recall[i], f1[i] ])
    
    header = [ 'Label ID', 'Label Name', 'Precision', 'Recall', 'F1']
    df = pd.DataFrame(data, columns=header)
    pd.set_option('display.max_rows', None)
    print(df.to_string(index=False, col_space=10))    
    return distance_matrix

def read_file(filename):
    featureDesc = torch.load(filename)
    return featureDesc

def get_data_to_store(file_data, labelled_data):
    data = []
    for imageid in file_data:
        if(imageid % 2 != 0):
            map = {
                "imageID": imageid,
                "label": labelled_data[imageid] or "unknown",
                "feature_descriptor": file_data[imageid].tolist()
            }
            data.append(map)  
    return data

def get_odd_iamges(model, labelled_data):
    # TODO: if PKL file is not present, compute it then and there
    file_data = read_file(model+"_vectors.pkl")
    data = get_data_to_store(file_data, labelled_data)
    return data
