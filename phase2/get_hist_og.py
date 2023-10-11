import cv2
import numpy as np
from utils import *

class histogram_of_oriented_gradients():
    def __init__(self) -> None:
        pass

    def magnitude_weighted_histogram(self,gradient_magnitude,gradient_orientation, grid):
        """This function distributes the magnitude in current angle bin and next bin 
        according to how close the orientation is to the bin"""
        number_of_bins = 9
        # As each bin is of 40 degrees so bin width is 40
        bin_width = 40
        hist_bins = np.arange(-180, 181, bin_width)
        # Initializing the grid histogram magnitude values will be added here based on the oeientaion
        grid_hist = np.zeros(9)
        # below code will calculate histogram by going to through each pixels orientation and magnitude
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                pixel_magnitude = gradient_magnitude[x, y]
                pixel_angle = gradient_orientation[x, y]

                # Getting the current bin index to understand in which bin the vector is
                current_bin_index = int((pixel_angle + 180) / bin_width)

                # This is calculating the percentage that the magnitude contributes to the current bin
                if current_bin_index == number_of_bins:
                    current_bin_index = 0  # Wrap around for 180 degrees
                    magnitude_contribution = ((pixel_angle - abs(hist_bins[current_bin_index])) / bin_width)
                else:
                    magnitude_contribution = ((pixel_angle - hist_bins[current_bin_index]) / bin_width)

                # Calculating and adding the magnitude to the current bin and then adding the remaining magnitude to the next bean
                grid_hist[current_bin_index] += (1 - magnitude_contribution) * pixel_magnitude
                next_bin_index = (current_bin_index + 1) % number_of_bins
                grid_hist[next_bin_index] += magnitude_contribution * pixel_magnitude

        return grid_hist

        
    def compute_hog(self, users_image):
        """This function computes HOG with <-1,0,1> and <-1,0,1>^t masks with 9 signed bins"""
        # Resizing to 300x100 pixels
        users_image = convert_image_to_grayscale(users_image)

        resized_image = cv2.resize(users_image, (300, 100))
        # 10x10 grid
        grid_rows, grid_cols = 10, 10

        image_HOG = np.zeros((grid_rows, grid_cols, 9), dtype=np.float32)

        # Define the masks for computing dI/dx and dI/dy gradients
        mask_x = np.array([[-1, 0, 1]])
        mask_y = np.array([[-1], [0], [1]])

        grid_height, grid_width = resized_image.shape[0] // grid_rows, resized_image.shape[1] // grid_cols

        # Below code creates 10x10 grid of the image and apply mask on the grid as well as calculate magnitude and orientation
        for i in range(grid_rows):
            for j in range(grid_cols):
                # Defining the coordinates of the current grid
                # Here x_start will be 0 , 30 , 60 .... and y_start will be 0, 10, 20 ...
                x_start = i * grid_height
                x_end = (i + 1) * grid_height
                y_start = j * grid_width
                y_end = (j + 1) * grid_width
                
                # Getting the current grid from the image
                grid = resized_image[x_start:x_end, y_start:y_end]

                # Applying the masks on the image to get horizontal and vertical gradients
                gradient_x = cv2.filter2D(grid, cv2.CV_64F,kernel= mask_x)
                gradient_y = cv2.filter2D(grid, cv2.CV_64F,kernel= mask_y)
                
                # Calculating the gradient magnitude and orientation
                gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                gradient_orientation = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)  

                # Calling the function defined above to calculate magnitude weighted histogram
                grid_hist = self.magnitude_weighted_histogram(gradient_magnitude, gradient_orientation, grid)

                # Store the histogram in the feature descriptor
                image_HOG[i, j] = grid_hist

        # For use cases where we require 1D array
        #image_HOG = image_HOG.reshape(-1)

        # print(image_HOG)
        return image_HOG


