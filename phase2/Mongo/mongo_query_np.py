from . import mongo_query
from .mongo_query import *
import numpy as np
from pymongo import MongoClient

def get_all_feature_descriptor_for_label(collection_name, label_name):
    collection = get_collection(collection_name)
    all_vectors = collection.find({"label": label_name})
    feature_descriptors = []
    
    for document in all_vectors:
        # image_ID = document.get("imageID")
        feature_descriptor = document['feature_descriptor']
        # feature_descriptors[image_ID] = np.array(feature_descriptor)
        feature_descriptors.append(np.array(feature_descriptor))
    return np.array(feature_descriptors)

# Function to get all feature descriptors in numpy array for a given collection name
def get_all_feature_descriptor(collection_name):
    collection  = get_collection(collection_name)

    document = collection.find_one({"imageID": 0})
    feature_descriptor = document.get("feature_descriptor")
    feature_descriptor = np.array(feature_descriptor)
    # feature_descriptors = np.zeros((count,) + feature_descriptor.shape)
    feature_descriptors = []

    all_vectors = collection.find()
    for document in all_vectors:
        # image_ID = document.get("imageID")
        feature_descriptor = document['feature_descriptor']
        # feature_descriptors[image_ID] = np.array(feature_descriptor)
        feature_descriptors.append(np.array(feature_descriptor))
    return np.array(feature_descriptors)

# Function to get the feature descriptor in numpy array of given imageID and collection name
def get_feature_descriptor(collection_name, imageID):

    collection = get_collection(collection_name)

    document = collection.find_one({"imageID": imageID})
    feature_descriptor = None
    try:
        feature_descriptor = document.get("feature_descriptor")
        feature_descriptor = np.array(feature_descriptor)
    except AttributeError:
        # Not in DB
        print('Lookup value not found in Database')

    return feature_descriptor


