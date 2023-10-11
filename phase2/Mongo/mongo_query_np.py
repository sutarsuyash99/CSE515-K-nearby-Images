from . import mongo_query
from .mongo_query import *
import numpy as np
from pymongo import MongoClient

# Function to get all feature descriptors in numpy array for a given collection name
def get_all_feature_descriptor(collection_name):
    collection  = get_collection(collection_name)

    count = collection.count_documents({})
    document = collection.find_one({"imageID": 0})
    feature_descriptor = document.get("feature_descriptor")
    feature_descriptor = np.array(feature_descriptor)
    feature_descriptors = np.zeros((count,) + feature_descriptor.shape)

    all_vectors = collection.find()
    
    for document in all_vectors:
        image_ID = document.get("imageID")
        feature_descriptor = document['feature_descriptor']
        feature_descriptors[image_ID] = np.array(feature_descriptor)
    return feature_descriptors

# Function to get the feature descriptor in numpy array of given imageID and collection name
def get_feature_descriptor(collection_name, imageID):

    collection = get_collection(collection_name)

    document = collection.find_one({"imageID": imageID})
    
    feature_descriptor = document.get("feature_descriptor")
    feature_descriptor = np.array(feature_descriptor)

    return feature_descriptor


