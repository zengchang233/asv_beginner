import yaml

def read_config(file_path): 
    f = open(file_path, 'r')
    config = yaml.load(f, Loader = yaml.CLoader)
    f.close()
    return config

def save_config():
    pass
