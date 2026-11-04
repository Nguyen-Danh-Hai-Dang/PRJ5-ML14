import math
import unittest

import torch
from losses import digitclassifier_loss


class DigitClassifierLossTest(unittest.TestCase):
    def test_uniform_logits_and_target_gradients(self):
        logits = torch.zeros(4, 10, requires_grad=True)
        labels = torch.eye(10)[:4].requires_grad_()
        loss = digitclassifier_loss(logits, labels)
        self.assertEqual(tuple(loss.shape), (1,))
        self.assertAlmostEqual(loss.item(), math.log(10), places=6)
        loss.backward()
        for gradient in (logits.grad, labels.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all().item())

    def test_correct_class_confidence_reduces_loss(self):
        labels = torch.eye(10)[:1]
        uniform = torch.zeros(1, 10)
        correct = uniform.clone()
        correct[0, 0] = 5.0
        wrong = uniform.clone()
        wrong[0, 1] = 5.0
        self.assertLess(digitclassifier_loss(correct, labels).item(),
                        digitclassifier_loss(uniform, labels).item())
        self.assertGreater(digitclassifier_loss(wrong, labels).item(),
                           digitclassifier_loss(uniform, labels).item())


if __name__ == "__main__":
    unittest.main()
