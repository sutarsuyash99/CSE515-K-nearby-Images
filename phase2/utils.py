import torchvision
from typing import Tuple

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
