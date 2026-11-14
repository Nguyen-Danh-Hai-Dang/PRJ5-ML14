import unittest

import torch
from torch.utils.data import Dataset

from train import train_digitclassifier


class ValidationDataset(Dataset):
    def __init__(self, model, accuracy):
        self.model = model
        self.accuracy = accuracy
        self.validation_states = []

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"x": torch.ones(2), "label": torch.eye(10)[0]}

    def get_validation_accuracy(self):
        if self.model.training or torch.is_grad_enabled():
            raise AssertionError("Validation must use eval mode without gradients")
        self.validation_states.append({
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        })
        return self.accuracy if len(self.validation_states) == 1 else 0.5


class DigitTrainingTest(unittest.TestCase):
    def check_training(self, trainer, accuracy, expected_epochs):
        model = torch.nn.Linear(2, 10)
        dataset = ValidationDataset(model, accuracy)
        trainer(model, dataset)
        self.assertEqual(len(dataset.validation_states), expected_epochs)
        self.assertFalse(model.training)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, dataset.validation_states[0][key])

    def test_q3_stops_with_earlier_qualifying_checkpoint(self):
        self.check_training(train_digitclassifier, 0.99, 5)

    def test_q3_respects_epoch_limit_below_threshold(self):
        self.check_training(train_digitclassifier, 0.6, 20)


if __name__ == "__main__":
    unittest.main()
