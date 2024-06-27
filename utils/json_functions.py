import json


def save_json(dict, path):
    '''
    Save dict as .json-file.
    '''
    with open(path, "w") as file:
        json.dump(dict, file)


def read_json(path):
    '''
    Read .json-file and returns dict.
    '''
    with open(path, "r") as file:
        dict = json.load(file)
    return dict