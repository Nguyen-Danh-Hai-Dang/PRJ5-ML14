import unittest

import torch
from models import DigitClassificationModel


class DigitClassificationModelTest(unittest.TestCase):
    def test_batch_shapes_and_gradients(self):
        for batch_size in (1, 4, 100):
            with self.subTest(batch_size=batch_size):
                model = DigitClassificationModel()
                images = torch.rand(batch_size, 784, requires_grad=True)
                logits = model(images)
                self.assertEqual(tuple(logits.shape), (batch_size, 10))
                self.assertTrue(torch.isfinite(logits).all().item())
                logits.square().mean().backward()
                self.assertIsNotNone(images.grad)
                self.assertTrue(torch.isfinite(images.grad).all().item())
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)


if __name__ == "__main__":
    unittest.main()
