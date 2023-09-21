from collections import defaultdict
import torchvision

dataset = torchvision.datasets.Caltech101(root='./data', download=True, target_type='category')

# this is going to be created once and passed throughout in all functions needed
# dict: (label: string, list<imageIds: int>)
# where key is either label index as int (eg: faces is with index 0) or it is the label name as string
# both are added to the map, user can decide which to use
# value is the list of ids belonging to that category
labelled_images = defaultdict(list)
dataset_named_categories = dataset.categories
for i in range(len(dataset)):
    img, label = dataset[i]
    # label returns the array index for dataset.categories
    labelled_images[str(label)].append(i)
    category_name = dataset_named_categories[label]
    labelled_images[category_name].append(i)
# print(labelled_images)