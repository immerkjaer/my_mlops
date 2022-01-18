# -*- coding: utf-8 -*-
import sys
import os
script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..' )
sys.path.append( mymodule_dir )

import logging
from pathlib import Path

import click
import numpy as np
import torch
from torchvision import transforms
from libs.utils import load_raw_data


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    loaded_data = load_raw_data(input_filepath)

    norm = transforms.Normalize((0.5,), (0.5,))

    tensor_train = norm(torch.from_numpy(loaded_data['images_train']).float())
    tensor_train_labels = torch.from_numpy(loaded_data['labels_train'])
    tensor_test = norm(torch.from_numpy(loaded_data['images_test']).float())
    tensor_test_labels = torch.from_numpy(loaded_data['labels_test'])

    torch.save(tensor_train, os.path.join(output_filepath, 'tensor_train.pt'))
    torch.save(tensor_train_labels, os.path.join(output_filepath, 'tensor_train_labels.pt'))
    torch.save(tensor_test, os.path.join(output_filepath, 'tensor_test.pt'))
    torch.save(tensor_test_labels, os.path.join(output_filepath, 'tensor_test_labels.pt'))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
