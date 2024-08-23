import pickle

def data_pickle(data, file_name, dir = 'representations_data\\'):
    with open(dir + file_name + '.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def unpickle(file_name, dir = 'representations_data\\'):
    with open(dir + file_name + '.pickle', 'rb') as f:
        return pickle.load(f)