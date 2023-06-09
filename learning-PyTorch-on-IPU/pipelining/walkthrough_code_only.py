# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

# THIS FILE IS AUTOGENERATED. Rerun SST after editing source file: walkthrough.py

# The following code is only necessary to prevent errors when running the source
# file as a script.


def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


if __name__ == "__main__" and not is_interactive():
    print(
        "This tutorial has been designed to run in a Jupyter notebook. "
        "If you would like to run it as a Python script, please "
        "use mnist_pipeline.py instead. This is required due to Python "
        "process spawning issues when using asynchronous data loading, "
        "as detailed in the README."
    )
    exit(0)

import os
import json
import argparse
from pkgutil import get_data
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import poptorch

learning_rate = 0.001
epochs = 3
batch_size = 40
test_batch_size = 8
training_iterations = 10
gradient_accumulation = 10
inference_iterations = 100

train_dataset = torchvision.datasets.MNIST(
    "mnist_data/",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

test_dataset = torchvision.datasets.MNIST(
    "mnist_data/",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

train_opts = poptorch.Options()
train_opts.deviceIterations(training_iterations)
train_opts.Training.gradientAccumulation(gradient_accumulation)

training_data = poptorch.DataLoader(
    train_opts,
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    mode=poptorch.DataLoaderMode.Async,
    num_workers=16,
)

test_data = poptorch.DataLoader(
    poptorch.Options().deviceIterations(inference_iterations),
    test_dataset,
    batch_size=test_batch_size,
    shuffle=True,
    drop_last=True,
    mode=poptorch.DataLoaderMode.Async,
    num_workers=16,
)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, pool_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        return x


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ConvLayer(1, 10, 5, 2)
        self.layer2 = ConvLayer(10, 20, 5, 2)
        self.layer3 = nn.Linear(320, 256)
        self.layer3_act = nn.ReLU()
        self.layer4 = nn.Linear(256, 10)

    def forward(self, x):
        with poptorch.Block("B1"):
            x = self.layer1(x)
        with poptorch.Block("B2"):
            x = self.layer2(x)
        with poptorch.Block("B3"):
            x = x.view(-1, 320)
            x = self.layer3_act(self.layer3(x))
        with poptorch.Block("B4"):
            x = self.layer4(x)
        return x


model = Network()


class TrainingModelWithLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, args, labels=None):
        output = self.model(args)
        if labels is None:
            return output
        with poptorch.Block("B4"):
            loss = self.loss(output, labels)
        return output, loss


def get_training_model(opts, model):
    """Wrap a model with the loss and the optimiser into a poptorch.trainingModel"""
    model_with_loss = TrainingModelWithLoss(model)
    training_model = poptorch.trainingModel(
        model_with_loss,
        opts,
        optimizer=poptorch.optim.AdamW(model.parameters(), lr=learning_rate),
    )
    return training_model


train_opts = poptorch.Options()

pipelined_strategy = poptorch.PipelinedExecution(
    poptorch.Stage("B1").ipu(0),
    poptorch.Stage("B2").ipu(1),
    poptorch.Stage("B3").ipu(2),
    poptorch.Stage("B4").ipu(3),
)

train_opts.setExecutionStrategy(pipelined_strategy)

train_opts.deviceIterations(training_iterations)
train_opts.Training.gradientAccumulation(gradient_accumulation)

sharded_strategy = poptorch.ShardedExecution(
    poptorch.Stage("B1").ipu(0),
    poptorch.Stage("B2").ipu(1),
    poptorch.Stage("B3").ipu(2),
    poptorch.Stage("B4").ipu(3),
)

train_opts.TensorLocations.setOptimizerLocation(poptorch.TensorLocationSettings().useOnChipStorage(False))


def train(training_model, training_data):
    nr_steps = len(training_data)
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        bar = tqdm(training_data, total=nr_steps)
        for data, labels in bar:
            preds, losses = training_model(data, labels)
            mean_loss = torch.mean(losses).item()
            acc = accuracy(preds, labels)
            bar.set_description(f"Loss:{mean_loss:0.4f} | Accuracy:{acc:0.2f}%")


def accuracy(predictions, labels):
    _, ind = torch.max(predictions, 1)
    labels = labels[-predictions.size()[0] :]
    accuracy = torch.sum(torch.eq(ind, labels)).item() / labels.size()[0] * 100.0
    return accuracy


# Warning: impacts performance
train_opts.outputMode(poptorch.OutputMode.All)

training_model = get_training_model(train_opts, model)
train(training_model, training_data)


def test(inference_model, test_data):
    nr_steps = len(test_data)
    sum_acc = 0.0
    for data, labels in tqdm(test_data, total=nr_steps):
        output = inference_model(data)
        sum_acc += accuracy(output, labels)
    print(f"Accuracy on test set: {sum_acc / len(test_data):0.2f}%")


inf_options = train_opts.clone()
inf_options.Training.gradientAccumulation(1)
inf_options.deviceIterations(inference_iterations)
inference_model = poptorch.inferenceModel(model, inf_options)

test(inference_model, test_data)

# Generated:2022-09-27T15:36 Source:walkthrough.py SST:0.0.8
