# DO NOT USE 
# Not implementing in project because of high latency

import numpy as np
import boto3
import tqdm
import torch
import json

from utils import img_label_and_named_label_for_query_int


# Function to load the feature descriptors along with label name to dynamodb table. Provide dataset, table name and pickle file name.
def pickle_to_dynamodb(dataset, table_name, filename):
    # Initialize AWS DynamoDB client
    dynamodb = boto3.client('dynamodb', region_name='us-east-2')

    # Load the data from pickle file
    featureDesc = torch.load(filename)
    feature_list = list(featureDesc.values())

    # Insert the feature descriptors and labelnames for all images to table
    for i in tqdm(range(len(dataset))):
        # Get the label name for the image
        img, label_id, label_name = img_label_and_named_label_for_query_int(dataset, i)
        feature_data = feature_list[i]  

        # Prepare the item for DynamoDB table
        item = {
            'ImageID': {'N': str(i)},  # Assuming 'ImageID' is a number
            'LabelName': {'S': label_name},  # Assuming 'LabelName' is a string
            'FeatureData': {'S': json.dumps(feature_data.tolist())},
        }

        # Insert the item into the DynamoDB table
        dynamodb.put_item(TableName=table_name, Item=item)



# Function to retrieve label name and feature descriptor from the dynamoDB table.
def get_features(table_name, imageID):
    # Initialize AWS DynamoDB client
    dynamodb = boto3.client('dynamodb', region_name='us-east-2')

    # Get the item from dynamoDB table
    response = dynamodb.get_item(
        TableName=table_name,
        Key={'ImageID': {'N': str(imageID)}}
    )

    # Extraction and typecasting of data to required datatypes 
    if 'Item' in response:
        item = response['Item']

        label_name = item['LabelName']['S']

        feature_data_json = item['FeatureData']['S']
        feature_data = np.array(json.loads(feature_data_json))
        return label_name, feature_data
    
    return None