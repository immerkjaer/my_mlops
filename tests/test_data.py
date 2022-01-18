import numpy as np
from numpy.lib.arraysetops import unique

from src.libs.utils import load_raw_data
from tests import _PATH_DATA_RAW

N_TRAIN = 40000
N_TEST = 5000
LABELS = list(range(0, 10))
IMAGE_SHAPE = (28, 28)

LOADED_DATA = load_raw_data(_PATH_DATA_RAW)

def test_data_length():
    
    assert(len(LOADED_DATA['images_train']) == N_TRAIN)
    assert(len(LOADED_DATA['images_test']) == N_TEST)
    
def test_image_shape():

    for img in LOADED_DATA['images_train']:
        assert(np.shape(img) == IMAGE_SHAPE)

    for img in LOADED_DATA['images_test']:
        assert(np.shape(img) == IMAGE_SHAPE)

def test_all_labels_represented():

    unique_labels_test = list(set(LOADED_DATA['labels_test']))
    unique_labels_train = list(set(LOADED_DATA['labels_train']))

    assert(sorted(unique_labels_test) == LABELS)
    assert(sorted(unique_labels_train) == LABELS)



