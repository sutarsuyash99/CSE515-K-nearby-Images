import torchvision
from typing import Tuple
from ordered_set import OrderedSet

# for query image id, return label name for it
def name_for_label_index(dataset: torchvision.datasets.Caltech101, index: int) -> str:
    dataset_named_categories = dataset.categories
    return dataset_named_categories[index]

# for query image id, return (PIL image, label_id, label_name) 
# returns IndexError if index is not in range
def img_label_and_named_label_for_query_int(dataset: torchvision.datasets.Caltech101, 
                                            index: int) -> Tuple[any, int, str]:
    if(index > len(dataset) or index < 0): 
        print("Not proper images")
        return IndexError
    else:
        img, label_id = dataset[index]
        label_name = name_for_label_index(dataset, label_id)
        return (img, label_id, label_name)
        
        




def label_feature_descriptor_fc(): 

    '''
    Loads data from pickle file that contains RESNET FC feature vector along with label id and name for each image. 
    Format of the pickle file : {'id': 8676, 'label_id': 100, 'label': 'yin_yang', 'ResnetFC': tensor([])}
    Total 101 labels from 0 to 100
    This is to be replaced by data from DB
    '''
    
    #Loading data
    fc_with_labels = torch.load('fc_labels.pkl')

    #Stores info of every label in form of a dictionary and appends all the entries to a list 
    list_of_label_dictionaries = []
    
    #Gets all the unique label names and the number of labels from the data in ordered format
    labels = list(OrderedSet([entry['label'] for entry in fc_with_labels]))
    number_of_labels = len(labels)
    
    #For all the images of a label_id create dictionary containing label_id, label and label feature descriptor
    #Append the dictionary in to a list 
    for label_id in range(number_of_labels):
        
        label_dict = {}
        
        #Creates a list of all the feature vectors for a particular label id
        label_feature_descriptors = [ entry['ResnetFC'].flatten() for entry in fc_with_labels if entry['label_id'] == label_id ]
        
        #Stacks into one tensor and takes mean to ouput a label feature descriptor
        label_super_descriptor = torch.stack(label_feature_descriptors).mean(0)
        
        #Create the dictionary with details about the label and append to list
        label_dict['label_id'] = label_id 
        label_dict['label'] = labels[label_id]
        label_dict['label_feature'] = label_super_descriptor
        
        list_of_label_dictionaries.append(label_dict)
    
    
    #Saving to a pkl file 
    torch.save(list_of_label_dictionaries,'fc_labels_mean.pkl')
    