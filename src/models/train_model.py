# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import numpy as np
import torch
# import wandb

from torch import nn, optim

# wandb.init()

@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath_figure", type=click.Path())
@click.argument("output_filepath_model", type=click.Path())
def main(input_filepath, output_filepath_figure, output_filepath_model):

    logger = logging.getLogger(__name__)
    logger.info("training model from processed data")

    # Loaded as tensors
    tensor_train = torch.load(os.path.join(input_filepath, "tensor_train.pt"))
    labels_train = torch.load(os.path.join(input_filepath, "tensor_train_labels.pt"))
    tensor_test = torch.load(os.path.join(input_filepath, "tensor_test.pt"))
    labels_test = torch.load(os.path.join(input_filepath, "tensor_test_labels.pt"))

    trainloader_data = torch.utils.data.TensorDataset(tensor_train, labels_train)
    testloader_data = torch.utils.data.TensorDataset(tensor_test, labels_test)

    trainloader = torch.utils.data.DataLoader(trainloader_data, batch_size=16)
    testloader = torch.utils.data.DataLoader(testloader_data, batch_size=16)

    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        nn.LogSoftmax(),
    )

    # wandb.watch(model, log_freq=100)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)

    epochs = 30
    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:

            # Flatten
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            test_loss = 0
            accuracy = 0

            with torch.no_grad():
                model.eval()
                for images, labels in testloader:
                    images = images.view(images.shape[0], -1)

                    log_ps = model(images)
                    test_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            model.train()

            train_losses.append(running_loss / len(trainloader))
            test_losses.append(test_loss / len(testloader))

            logger.info("Epoch: {}/{}.. ".format(e + 1, epochs))
            logger.info("Training Loss: {:.3f}.. ".format(train_losses[-1]))
            logger.info("Test Loss: {:.3f}.. ".format(test_losses[-1]))
            logger.info("Test Accuracy: {:.3f}".format(accuracy / len(testloader)))

            # wandb.log({"Epoch number": e+1})
            # wandb.log({"Training Loss": train_losses[-1]})
            # wandb.log({"Test Loss": test_losses[-1]})
            # wandb.log({"Test Accuracy": accuracy / len(testloader)})

    # plt.plot(train_losses, label="Training loss")
    # plt.plot(test_losses, label="Validation loss")
    # plt.savefig(os.path.join(output_filepath_figure, "foo.png"))
    # plt.clf()

    # wandb.log({"my_custom_plot_id": wandb.plot.line_series(
    #     xs=list(range(1,epochs+1)),
    #     ys=[[float(x) for x in train_losses], [float(x) for x in test_losses]],
    #     keys=["train losses", "test losses"],
    #     title="Train loss VS test loss",
    #     xname="epochs"
    #     )})
    
    torch.save(model.state_dict(), os.path.join(output_filepath_model, "model.pth"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]


    main()
