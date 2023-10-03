import mongo_connection
from bson.json_util import dumps

def query(collection_name, imageID):
    db = mongo_connection.get_database()
    collection = db[collection_name]
    cursor = collection.find({"imageID": imageID})
    list_cur = list(cursor)
    json_data = dumps(list_cur, indent = 2) 
    print(json_data)
    return json_data