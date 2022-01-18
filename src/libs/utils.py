import os
import numpy as np

def load_raw_data(input_filepath: str) -> dict[str, np.ndarray]:
    imagesTrain = []
    labelsTrain = []

    imagesTest = []
    labelsTest = []

    loaded_data = {}

    directory = os.fsencode(input_filepath)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".npz"): 
            with np.load(os.path.join(input_filepath, filename)) as f:
                if 'test' not in filename:
                    imagesTrain.append(f['images'])
                    labelsTrain.append(f['labels'])
                else:
                    imagesTest.append(f['images'])
                    labelsTest.append(f['labels'])

    loaded_data['images_train'] = np.concatenate(imagesTrain)
    loaded_data['labels_train'] = np.concatenate(labelsTrain)

    # Not stable if more than 1 test file...
    loaded_data['images_test'] = np.array(imagesTest[0])
    loaded_data['labels_test'] = np.array(labelsTest[0])

    return loaded_data