import os
_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data
_PATH_DATA_RAW = os.path.join(_PATH_DATA, "raw") 
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")