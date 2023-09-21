from utils import display_image_og
from PIL import ImageOps, Image
import numpy as np
import cv2
from matplotlib import pyplot

# displaying gradient subplots after convolve application of masks
def show_subplots(gradient_x, gradient_y):
    fig, (ax1, ax2) = pyplot.subplots(2,1)
    ax1.imshow(gradient_x, cmap = 'gray')
    ax1.set_title('Gradient x')
    ax2.imshow(gradient_y, cmap = 'gray')
    ax2.set_title('Gradient y')
    fig.show()

def print_features(histograms):
    print(f"There are {histograms.size} features in the following shape {histograms.shape}")
    with np.printoptions(threshold=np.inf):
        print(histograms)
    print("*"*30)
    print("\n\n\n")


def compute_hog(pil_img, new_size: tuple, in_bulk: bool = False):
    pil_img = ImageOps.grayscale(pil_img)
    pil_img = display_image_og(pil_img=pil_img, new_size=new_size, in_bulk=in_bulk)

    # convert image to numpy array
    gray_image = np.array(pil_img)
    h, w = gray_image.shape
    cell_size = (30, 10)
    # print(h, w, gray_image.shape)

    # compute gradients using masks
    mask_x, mask_y = np.array([[-1,0,1]]) , np.array([[-1],[0],[1]])

    gradient_x = cv2.filter2D(gray_image, cv2.CV_64F, kernel=mask_x)
    gradient_y = cv2.filter2D(gray_image, cv2.CV_64F, kernel=mask_y)

    if(not in_bulk): show_subplots(gradient_x=gradient_x, gradient_y=gradient_y)

    # compute magnitude and angle
    magnitude = np.sqrt((gradient_x) ** 2 + gradient_y**2)
    angle = np.arctan2(gradient_y,gradient_x) * (180/np.pi)

    # print(angle.shape, min(angle.flatten()), max(angle.flatten()))

    num_bins = 9
    bin_width = 40
    histograms = np.zeros((h // cell_size[1], w // cell_size[0], num_bins))
    bins_matching_angles = np.array(range(-180, 180+1, 40))
    # print(histograms.shape)

    for i in range(0, h, cell_size[1]):
        for j in range(0, w, cell_size[0]):
            cur_magnitude = magnitude[i:i+cell_size[1], j:j+cell_size[0]]
            cur_angle = angle[i:i+cell_size[1], j:j+cell_size[0]]

            cell_histograms = np.zeros(num_bins)
            for ii in range(0, 10):
                for jj in range(0, 30):
                    cur_cur_angle = cur_angle[ii, jj]
                    #                           0   1       2      3    4       5   6       7   8   9
                    # bins_matching_angles = [-180, -140, -100,  -60,  -20,   20,   60,  100,  140, 180]
                    if(cur_cur_angle in bins_matching_angles):
                        # add to that bin only
                        cur_bin = np.where(bins_matching_angles == cur_cur_angle)[0][0] % 9
                        cell_histograms[cur_bin] += cur_magnitude[ii][jj]
                    else:
                        # find lower, higher angles
                        # target is cur_cur_angle
                        try:
                            lower_matching_angle = bins_matching_angles[bins_matching_angles < cur_cur_angle].max()
                            higher_matching_angle = bins_matching_angles[bins_matching_angles > cur_cur_angle].min()
                        except ValueError:
                            pass

                        lower_bin = np.where(bins_matching_angles == lower_matching_angle)[0][0] % 9
                        higher_bin = np.where(bins_matching_angles == higher_matching_angle)[0][0] % 9

                        cell_histograms[lower_bin] += (np.abs(higher_matching_angle - cur_cur_angle)/bin_width) * (cur_magnitude[ii][jj])
                        cell_histograms[higher_bin] += (np.abs(cur_cur_angle - lower_matching_angle)/bin_width) * (cur_magnitude[ii][jj])
            # in inner loops
            histograms[i // cell_size[1], j // cell_size[0], :] = cell_histograms

    if(not in_bulk): print_features(histograms=histograms)
    return histograms