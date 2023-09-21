from color_moments import rgb_color_moments
from hog import compute_hog
from resnet_50 import resnet_50_init

def individual_img(pil_img):

    print("Choose any of the following model run feature models:\
          \n\n\
          1 -> Color moments\
          \n\
          2 -> Histogram of gradient\
          \n\
          3 -> RESNET-50\
          \n\
          4 -> Quit\
          \n\n")
    choose = int(input())
    match choose:
        case 1: rgb_color_moments(pil_img=pil_img, new_size=(300, 100))
        case 2: compute_hog(pil_img=pil_img, new_size=(300,100))
        case 3: resnet_50_init(pil_img=pil_img, new_size=(224, 224))
        case 4: print("Exiting....")
        case default: print("Wrong Input")
