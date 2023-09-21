import torchvision
from individual_photo import individual_img
from feature_extract_pickle import compute_all_feature_extract_pickle
from searchK import compute_searchK

option = 1

def individual_img_input(dataset): 
    print("Please enter an Integer ID for image ranking")
    try:
        i = int(input())
    except ValueError:
        i = 10000
    
    if i < len(dataset):
        (img, _) = dataset[i]
        individual_img(img)
    else:
        print("Error input was provided")


while(option != 0):
    dataset = torchvision.datasets.Caltech101(root='./data', download=True)
    print("Enter corresponding input to run some task:\
        \n\
        1 -> Open individual image and run corresponding feature descriptor with input feature model\
        \n\
        2 -> Compute and Store all features for all images\
        \n\
        3 -> Searching K nearest images from given Image ids\
        \n\
        0 -> quit\
        \n\n")
    try:
        option = int(input())
    except ValueError:
        # enter the default case
        option = 4
    match option:
        case 0: print("Exiting....")
        case 1: individual_img_input(dataset)
        case 2: compute_all_feature_extract_pickle(dataset=dataset)
        case 3: compute_searchK(dataset)
        case default:  print("Wrong input provided")
        