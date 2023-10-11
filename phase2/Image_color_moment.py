import numpy as np
from  torchvision import datasets
import matplotlib.pyplot as plt
import cv2
from utils import * 


class color_moments():
    def __init__(self) -> None:
        pass  

    def get_skewness(self, grid, grid_mean, grid_stddev):
        """Calculate the skewness using the grid_mean and and grid pixel values"""

        # Calculating individul channel summation of the differences of the Grid pixel value and grid mean value
        temp_r = np.sum(((grid[:,:,0]-grid_mean[0])** 3))
        temp_g = np.sum(((grid[:,:,1]-grid_mean[1])** 3))
        temp_b = np.sum(((grid[:,:,2]-grid_mean[2])** 3))

        # Dividing with the number of pixels 30*10
        grid_skew_r = (temp_r)/300
        grid_skew_g = (temp_g)/300
        grid_skew_b = (temp_b)/300

        # Taking cube root but as numpy gives nan when taking cube root we are taking cube root of absolute 
        # values and then multiplying with the sign
        grid_skewness_r = np.sign(grid_skew_r)*((np.abs(grid_skew_r)) ** (1/3))
        grid_skewness_g = np.sign(grid_skew_g)*((np.abs(grid_skew_g)) ** (1/3))
        grid_skewness_b = np.sign(grid_skew_b)*((np.abs(grid_skew_b)) ** (1/3))
        grid_skew = [grid_skewness_r, grid_skewness_g, grid_skewness_b]

        return grid_skew
    
    def color_moments_fn(self, users_image):
        """This function calculates color moments for of an RGB image
        Color moments being Grid mean, Grid Standard Deviation and Grid Skewness."""
      
        # Converting the Image to opencv/numpy format
        users_image = check_rgb_change_grayscale_to_rgb(users_image)

        rgb_image = np.array(users_image)
        # cv2_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) 
        
        # We need to divide the image in the grid of 10x10
        rows = 10
        cols = 10
        image_rgb = cv2.resize(rgb_image,(300,100))
        # if rgb_image is not None:
        #     # Display the image in a window
        #     cv2.imshow('Image', cv2_image)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # else:
        #     print('Image not found or could not be loaded.')

        # Calculate the size of each grid
        height, width, _ = image_rgb.shape
        grid_height = height // rows
        grid_width = width // cols

        # Initialize lists to store color moments for each grid
        grid_means = []
        grid_stddevs = []
        grid_skewness = []

        for i in range(rows):
            for j in range(cols):
                # Defining the coordinates of the current grid
                # Here x_start will be 0 , 30 , 60 .... and y_start will be 0, 10, 20 ...
                x_start = i * grid_height
                x_end = (i + 1) * grid_height
                y_start = j * grid_width
                y_end = (j + 1) * grid_width
                
                # Extract the current grid from the image
                grid = image_rgb[x_start:x_end, y_start:y_end, :]
                
                # Calculate color moments for the current grid
                grid_mean = np.mean(grid, axis=(0, 1))
                grid_stddev = np.std(grid, axis=(0, 1))
                
                grid_skew = self.get_skewness(grid, grid_mean, grid_stddev)
                # print(grid_mean,grid_stddev,grid_skew)
                
                # Append the color moments for the current grid to the lists
                
                grid_means.append(grid_mean)
                grid_stddevs.append(grid_stddev)
                grid_skewness.append(grid_skew)

        # Converting all to numpy arrays 
        grid_means = np.array(grid_means)
        grid_stddevs = np.array(grid_stddevs)
        grid_skewness = np.array(grid_skewness) 

        # here the final color moments vector is 3d vector where first three values are Cell mean R G B 
        # then grid stddev  R G B and then Grid Skew with RGB values. 
        # EG - final_color_moments[0][0][0] will get us 1st grid's r channel mean. (Shape = 100 3 3)
        final_color_moments = np.stack([grid_means, grid_stddevs, grid_skewness], axis =1)
       
        # In case if we want the output in json
        final_output_json = {"color_moment_mean" : grid_means,
                        "color_moment_stddev" : grid_stddevs,
                        "color_moment_skewness" : grid_skewness
                        }
        # print(final_output)       
        return final_color_moments

