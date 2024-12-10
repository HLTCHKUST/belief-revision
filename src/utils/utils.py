import os


def is_model_on_api(model_name):
    
    api_model_list = ['gpt-3.5', 'gpt-4']
    
    is_model_on_api = False
    for api_model in api_model_list:
        if api_model in model_name:
            is_model_on_api = True
            
    return is_model_on_api


def prepare_paths(filepath):

    checks = filepath.split('/')[:-1]

    path = ''
    for dirpath in checks:
        path = os.path.join(path, dirpath)
        if not os.path.isdir(path):
            os.mkdir(path)