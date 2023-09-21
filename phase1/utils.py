from PIL import Image
import numpy as np
import math
from matplotlib import pyplot
from torchvision import transforms, datasets

def find_nearest_square(k: int) -> int:
    return math.ceil(math.sqrt(k))

def gen_unique_number_from_title(string: str) -> int:
    a = 0
    for c in string:
        a+=ord(c)
    return a

def display_k_images_subplots(dataset: datasets.Caltech101, distances: tuple, title: str):
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

def display_image(pil_img, new_size: tuple, in_bulk: bool = False)->Image:
    # resize image into new size and show image
    pil_img = pil_img.resize(new_size)
    # For any image, with less than 3 channels, we convert the image PIL mode to RGB and continue with analysis
    if(pil_img.mode != 'RGB'): pil_img = pil_img.convert("RGB")
    # these parameters are just to skip while pickling
    if(not(in_bulk)): pil_img.show()
    return pil_img

def display_image_og(pil_img, new_size: tuple, in_bulk: bool = False)->Image:
    # resize image into new size and show image
    pil_img = pil_img.resize(new_size)
    if(not(in_bulk)): pil_img.show()
    return pil_img

# I initially, thought there will be a lot of other work related to do, used only in color moments
def convert_pil_img_to_rgb_arrays(image: Image):
    return np.array(image)

# used only in color moments for the entire image, we can work on improving it
def display_histogram(array):
    image_r, image_g, image_b = array[:,:,0], array[:,:,1], array[:,:,2]
    # -0.5 is added as extra space
    bins = np.arange(-0.5, 255+1, 1)
    pyplot.hist(image_r.flatten(), bins = bins, color='r')
    pyplot.hist(image_g.flatten(), bins=bins, color='g')
    pyplot.hist(image_b.flatten(), bins=bins, color='b')
    pyplot.show(block=False)
    return pyplot

def segment_img_array(array, x_split_int: int, y_split_int: int):
    arr = []
    x_split_array = np.split(array, x_split_int)
    for x_grid in x_split_array:
        xy_split_array = np.split(x_grid, y_split_int, axis=1)
        for grid in xy_split_array:
           arr.append(grid)

    return arr 

def compute_mean_sd_skew(img_grid):
    means = np.mean(img_grid, axis=(0,1))
    std = np.std(img_grid, axis=(0,1))
    # The following skewness, kurtosis give out a bunch of nans and there is catastrophic calcution 
    # issue when all values are the same
    skew = compute_skew(img_grid=img_grid, mean=means)
    return [means, std, skew]

# defined a function for this, because we needed to compute skewness according to formula provided
def compute_skew(img_grid, mean):
    w,h,_ = img_grid.shape
    skew_r = np.sum((img_grid[:,:,0] - mean[0]) ** 3) / (w*h)
    skew_g = np.sum((img_grid[:,:,1] - mean[1]) ** 3) / (w*h)
    skew_b = np.sum((img_grid[:,:,2] - mean[2]) ** 3) / (w*h)

    # this is done to avoid numpy and python from freaking out about negative cube roots
    skew_r = np.sign(skew_r) * ((np.abs(skew_r)) ** (1/3))
    skew_g = np.sign(skew_g) * ((np.abs(skew_g)) ** (1/3))
    skew_b = np.sign(skew_b) * ((np.abs(skew_b)) ** (1/3))

    return [skew_r, skew_g, skew_b]

# used only in resnet computations
# https://stats.stackexchange.com/questions/458579/should-i-normalize-all-data-prior-feeding-the-neural-network-models
def convert_pil_tensor(pil_img: Image):
    transformer = transforms.Compose([transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
    img_tensor = transformer(pil_img)
    return img_tensor

# initially, I was going to ignore non 3 channel images, function not used anymore
def check_if_three_channels_for_image(pil_img):
    array = np.asarray(pil_img, dtype=np.float32)
    a = array.ndim
    print(a)
    if a <= 2:
        print('Skipping image as there no color channels')
        return False
    else: return True