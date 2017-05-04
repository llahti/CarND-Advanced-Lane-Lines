import os
def get_module_path():
    """
    This function returns absolute path of module. In otherword the 
    location of __init__.py file.
    """
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    return dir_path