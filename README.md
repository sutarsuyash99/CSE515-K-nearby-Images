# CSE515: Multimedia and web databases

> The project course follows feature extraction, feature engineering, CNNs

The folder structure will be divided into 3 folders `phase1`, `phase2`, `phase3` each having their own structures

***(Each sub-folder will have README inside of them)***

## Phase 1:
### Visual Information Extraction Project Readme

Experiment with image features, vector models, and similarity measures. Perform individual tasks using Python, PyTorch, and torchvision. Use ResNet50 for feature extraction on the Caltech 101 dataset. Tasks include visualizing images, extracting feature descriptors (Color moments, HOG, ResNet variants), storing descriptors, and visualizing similar images.

Project Tasks:

**Task 0:** Familiarize with Python, PyTorch, and torchvision. Download ResNet50 and Caltech 101 dataset. Choose storage method.
**Task 1:** Implement a program to visualize and extract feature descriptors:
Color moments (CM10x10)
Histograms of oriented gradients (HOG)
ResNet-AvgPool-1024
ResNet-Layer3-1024
ResNet-FC-1000
**Task 2:** Implement a program to extract and store feature descriptors for all images in the dataset.
**Task 3:** Implement a program to visualize the most similar k images based on each visual model. Use appropriate distance/similarity measures and list corresponding scores.

>> **NOTE:** Requires understanding of Color Moments(RGB), Histogram of oriented histograms(HOG), ResNet-50

## Phase-2:
### Multimodal Feature Analysis Project Readme

Explore image features, vector models, dimensionality, and graph analysis. Tasks involve using RESNET50 for feature extraction, image similarity, relevant label extraction, and latent semantic analysis. Implement dimensionality reduction techniques and create a similarity graph with personalized PageRank.

**Task 0:** Extract features with RESNET50, visualize similar images.
**Task 1:** Visualize relevant images for a given label.
**Task 2:** List likely matching labels for a query image.
**Tasks 3-6 (LS1-LS4):** Extract and store latent semantics.
**Tasks 7-10:** Visualize similar images, list matching labels under selected latent space.
**Task 11:** Create a similarity graph, identify significant images with personalized PageRank.


## Phase-3:
### Clustering, Indexing and Classification

Experiment with clustering, indexing, and classification/relevance feedback using the Caltech 101 dataset. Tasks involve feature models, similarity/distance functions, and latent space extraction algorithms developed in the previous phase. 

**Task 1:** For each unique label, compute k latent semantics for even-numbered Caltec101 images. Predict most likely labels for odd-numbered images using label-specific latent semantics. Output per-label precision, recall, F1-score, and overall accuracy.
**Task 2:** For each unique label, compute c most significant clusters with DBScan for even-numbered Caltec101 images. Visualize clusters in a 2D MDS space and as groups of image thumbnails. Predict most likely labels for odd-numbered images using label-specific clusters. Output per-label precision, recall, F1-score, and overall accuracy.
**Task 4:** Create m-NN classifier, decision-tree classifier, and PPR-based classifier for even-numbered Caltec101 images. Predict most likely labels for odd-numbered images using the user-selected classifier. Output per-label precision, recall, F1-score, and overall accuracy.
**Task 5:** Implement Locality Sensitive Hashing (LSH) tool for Euclidean distance. Perform a similar image search using the LSH index structure storing even-numbered Caltec101 images and a user-selected visual model. Visualize t most similar images and output unique and overall numbers of images considered.
**Task 6:** Implement an SVM-based and probabilistic relevance feedback system. Enable the user to tag results from Task 5b and return a new set of ranked results by revising the query or re-ordering existing results based on the selected feedback system.
