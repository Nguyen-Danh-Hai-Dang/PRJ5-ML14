from torch import no_grad
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model, dataset):
    with no_grad():
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True
        )

        changed = True

        while changed:
            changed = False

            for data in dataloader:
                x = data["x"].reshape(1, -1)
                y = data["label"].item()

                prediction = model.get_prediction(x)

                if prediction != y:
                    model.w += y * x
                    changed = True


def train_regression(model, dataset):
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for epoch in range(5000):
        model.train()

        for batch in dataloader:
            x = batch["x"]
            y = batch["label"]

            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)

            optimizer.zero_grad()

            y_pred = model(x)
            loss = regression_loss(y_pred, y)

            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            data_x = torch.tensor(
                dataset.x,
                dtype=torch.float32
            ).reshape(-1, 1)

            labels = torch.tensor(
                dataset.y,
                dtype=torch.float32
            ).reshape(-1, 1)

            train_loss = regression_loss(
                model(data_x),
                labels
            ).item()

        if train_loss <= 0.02:
            break


def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()
    """ YOUR CODE HERE """


def train_languageid(model, dataset):
    """
    Trains the model.

    Note that when you iterate through dataloader, each batch will returned as its own vector in the form
    (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
    get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
    that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
    as follows:

    movedim(input_vector, initial_dimension_position, final_dimension_position)

    For more information, look at the pytorch documentation of torch.movedim()
    """
    model.train()
    "*** YOUR CODE HERE ***"



def Train_DigitConvolution(model, dataset):
    model.train()
    dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for epoch in range(20):
        model.train()

        dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for epoch in range(20):
        model.train()

        dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for epoch in range(20):
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True
        )

        optimizer = optim.Adam(
            model.parameters(),
            lr=0.001
        )

    for epoch in range(20):
        model.train()

        for batch in dataloader:
            x = movedim(batch["x"], 0, 1)
            y = batch["label"]

            optimizer.zero_grad()

            y_pred = model(x)

            loss = languageid_loss(y_pred, y)

            loss.backward()

            optimizer.step()