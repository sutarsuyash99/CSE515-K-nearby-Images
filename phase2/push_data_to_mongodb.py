import mongo_connection
import pickle
import main

def read_file(filename):
    pickled_data = open(filename, 'rb')   
    filedata = pickle.load(pickled_data)
    print(len(filedata))
    pickled_data.close()
    return filedata

def get_data_to_store(file_data, labelled_data):
    data = []
    for imageid in file_data:
        map = {
            "imageID": imageid,
            "label": labelled_data[imageid] or "unknown",
            "feature_descriptor": file_data[imageid]
        }
        data.append(map)  
    return data

def upsert_data(collection_name, data):
    db = mongo_connection.get_database()
    collection = db[collection_name]
    for d in data:
        result = collection.insert_one(d)
        print("Inserted document ID:", result.inserted_id)

def process():
    file_data = read_file("resnet_avg_pool_output_file_v2")
    labelled_data = main.get_labelled_images()
    data = get_data_to_store(file_data, labelled_data)
    upsert_data("avgpool", data)
    
process()
