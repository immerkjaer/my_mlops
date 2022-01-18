import os 

_FILE_PATH = os.path.dirname(__file__)  # root of file (models)
_SRC_PATH = os.path.dirname(_FILE_PATH)  # root of models (src)
_PROJECT_PATH = os.path.dirname(_SRC_PATH) 

_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data
_RAW_DATA = os.path.join(_PATH_DATA, "raw") 
_PROCESSED_DATA = os.path.join(_PATH_DATA, "processed") 
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")