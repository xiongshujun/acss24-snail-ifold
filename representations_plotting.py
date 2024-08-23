import pickle

def unpickle(file_name):
    with open('representations_data\\' + file_name + '.pickle', 'rb') as f:
        return pickle.load(f)
    
