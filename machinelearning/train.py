import torch
from torch import no_grad
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from losses import languageid_loss

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
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_accuracy = -1.0
    best_state = None
    for epoch in range(20):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            loss = digitclassifier_loss(model(batch['x']), batch['label'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch['x'])
        model.eval()
        with no_grad():
            accuracy = dataset.get_validation_accuracy()
        print(f'Q3 epoch {epoch + 1}: loss={total_loss / len(dataset):.4f}, validation={accuracy:.2%}')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch >= 4 and accuracy >= 0.98:
            break
    model.load_state_dict(best_state)
    model.eval()



def train_languageid(model, dataset):
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.003)
    batch_size = 32

    while True:
        model.train()

        for x, y in dataset.iterate_once(batch_size):
            optimizer.zero_grad()

            logits = model(x)

            loss = languageid_loss(logits, y)
            loss.backward()
            optimizer.step()

        val_acc = dataset.get_validation_accuracy()
        print(f"Validation Accuracy: {val_acc * 100:.2f}%")

        if val_acc >= 0.84:
            break

    return model


def Train_DigitConvolution(model, dataset):
    dataloader = DataLoader(dataset, batch_size=25, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_accuracy = -1.0
    best_state = None
    for epoch in range(30):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            loss = digitconvolution_Loss(model(batch['x']), batch['label'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch['x'])
        model.eval()
        with no_grad():
            accuracy = dataset.get_validation_accuracy()
        print(f'Q5 epoch {epoch + 1}: loss={total_loss / len(dataset):.4f}, validation={accuracy:.2%}')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch >= 4 and accuracy >= 0.96:
            break
    model.load_state_dict(best_state)
    model.eval()
