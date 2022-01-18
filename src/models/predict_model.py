# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import torch
from torch import nn


@click.command()
@click.argument("input_filepath_model", type=click.Path(exists=True))
@click.argument("input_filepath_test", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath_model, input_filepath_test, output_filepath):

    logger = logging.getLogger(__name__)
    logger.info("predict")

    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        nn.LogSoftmax(),
    )

    tensor_test = torch.load(os.path.join(input_filepath_test, "tensor_test.pt"))
    labels_test = torch.load(os.path.join(input_filepath_test, "tensor_test_labels.pt"))

    testloader_data = torch.utils.data.TensorDataset(tensor_test, labels_test)

    testloader = torch.utils.data.DataLoader(testloader_data, batch_size=1024)

    state_dict = torch.load(input_filepath_model)
    model.load_state_dict(state_dict)

    images, labels = next(iter(testloader))
    images = images.view(images.shape[0], -1)
    ps = torch.exp(model(images))
    top_p, top_class = ps.topk(1, dim=1)

    equals = top_class == labels.view(*top_class.shape)

    accuracy = torch.mean(equals.type(torch.FloatTensor))
    logger.info("Accuracy: {:.3f}".format(accuracy.item() * 100))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
