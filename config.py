import os.path

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def j(rel_path):
    return os.path.join(PROJECT_DIR, rel_path)

def dim():
    return 1024 
