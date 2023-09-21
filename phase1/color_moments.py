from utils import display_image, convert_pil_img_to_rgb_arrays, display_histogram, segment_img_array, compute_mean_sd_skew, check_if_three_channels_for_image
import numpy as np

def print_color_moments_feature_vectors(grid_number: int, summary_info):
    print("For Grid number: ", grid_number)
    print("\tMean in form RGB: ", summary_info[grid_number][0])
    print("\tS.D. in form RGB: ", summary_info[grid_number][1])
    print("\tSkewness in form RGB: ", summary_info[grid_number][2])
    print("---------------------------------------------------------")

def printing_all(summary_info):
    print(" ----------------------- Printing all --------------------------- ")
    for i in range(len(summary_info)):
        print_color_moments_feature_vectors(grid_number=i, summary_info=summary_info)
    print(" ----------------------- Printing done --------------------------- ")

def rgb_color_moments(pil_img, new_size: tuple, in_bulk: bool = False):
    image = display_image(pil_img, new_size=new_size, in_bulk=in_bulk)
    # convert PIL to numpy array RGB
    rgb_data = convert_pil_img_to_rgb_arrays(image=image)
    # print(f'Image features: {rgb_data.size}')
    
    # display histogram over the entire image
    if(not in_bulk): display_histogram(array=rgb_data)
    
    # enumerate over grids 
    # splitting into grids
    # print(f"Image params: {rgb_data.shape}")
    img_grid = segment_img_array(array=rgb_data, x_split_int=10, y_split_int=10)
    # print(f"The image has been segmented into {len(img_grid)} grids")
    
    # over channels compute with mean, sd, skewness
    summary_info = []
    for grid in img_grid:
        summary_info.append(compute_mean_sd_skew(grid))

    if(not(in_bulk)): printing_all(summary_info=summary_info)
    return summary_info