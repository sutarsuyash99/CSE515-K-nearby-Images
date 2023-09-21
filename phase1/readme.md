# Phase 1: Multimedia and web databases 

***NOTE:*** This readme delves more into first phase of project

**AIM:**
The aim of this phase can be broken down into 3 parts:

***INFO:*** Dataset used here is `Caltech_101` from `torchvision`

1. Task1: Implement a program which, given an image ID and one of the following feature models, visualizes the image and then extracts and prints (in a human readable form) the corresponding feature descriptors:
    1. Color Moments
    2. Histograms of oriented gradients
    3. RESNET-50
        1. Avgpool-1024
        2. Layer3-1024
        3. FC-1000
2. Implement a program which extracts and stores feature descriptors for all the images in the data set.
3. Implement a program which, given an image ID and a value “k”, returns and visualizes the most similar k images based on each of the visual model you will select the appropriate distance/similarity measure fo each feature model. For each 0match, also list the corresponding distance/similarity score.

## Installation Steps:

### Pre-installation requirements:

1. `python3` installed and working (Recommended version 3.11.4 -- this is the version on author's machine)
2. First time installation on a new machine requires Internet connection -- downloading libraries and artifacts

### Configuration Steps:

1. Run `. startup.sh` in the subfolder containing that file
(Somethings `bash startup.sh` works better) 


## Project Structure

**Info:** There are multiple files not pushed and ignored in `.gitignore` file. We will go over how to retrieve those files

1. `Main.py`: this is the entry file to run to complete all the actions needed to achieve above tasks
2. `startup.sh`: This file is to be run to create a `.venv` and install the dependencies required to run the project
3. `requirements.txt`: Contains list of all the dependencies required to run this project
4. `**.pkl`: These files not be available on Github, these contain the trained feature vectors. Due to github's size limit, these won't be present on remote, but running main.py provides an option to regenerate them (it takes some time, grab a coffee!)
5. All the other files are submodules called internally by main.py based on the input given
6. Project Submission will also contain submission related output and report files not available here!

## Sample working terminal example:

@TODO: Add image here