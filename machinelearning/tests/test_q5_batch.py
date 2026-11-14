import unittest

import torch
from torch.nn.functional import conv2d

from models import Convolve, DigitConvolutionalModel


class ConvolveBatchTest(unittest.TestCase):
    def test_single_and_batched_values(self):
        generator = torch.Generator().manual_seed(5)
        for shape in ((5, 7), (1, 5, 7), (4, 5, 7)):
            for kernel_shape in ((1, 1), (2, 3), (5, 7)):
                with self.subTest(shape=shape, kernel=kernel_shape):
                    image = torch.randn(*shape, generator=generator, dtype=torch.double)
                    kernel = torch.randn(*kernel_shape, generator=generator, dtype=torch.double)
                    batch = image.reshape(-1, 1, 5, 7)
                    expected = conv2d(batch, kernel[None, None])
                    expected = expected.reshape(*shape[:-2], *expected.shape[-2:])
                    torch.testing.assert_close(Convolve(image, kernel), expected)

    def test_batched_gradients(self):
        generator = torch.Generator().manual_seed(6)
        image = torch.randn(4, 5, 7, generator=generator, dtype=torch.double, requires_grad=True)
        kernel = torch.randn(2, 3, generator=generator, dtype=torch.double, requires_grad=True)
        actual = Convolve(image, kernel)
        expected = conv2d(image[:, None], kernel[None, None])[:, 0]
        weights = torch.randn(actual.shape, generator=generator, dtype=torch.double)
        actual_gradients = torch.autograd.grad((actual * weights).sum(), (image, kernel))
        expected_gradients = torch.autograd.grad((expected * weights).sum(), (image, kernel))
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_noncontiguous_input(self):
        image = torch.arange(70, dtype=torch.double).reshape(2, 7, 5).transpose(-1, -2)
        kernel = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.double)
        expected = conv2d(image[:, None], kernel[None, None])[:, 0]
        torch.testing.assert_close(Convolve(image, kernel), expected)


    def test_model_values_and_gradients(self):
        model = DigitConvolutionalModel().double()
        images = torch.rand(4, 784, dtype=torch.double, requires_grad=True)
        actual = model(images)
        features = conv2d(
            images.reshape(4, 1, 28, 28),
            model.convolution_weights[None, None],
        ).flatten(start_dim=1)
        expected = model.output(model.layer1(features).relu())
        torch.testing.assert_close(actual, expected)
        inputs = (images, *model.parameters())
        actual_gradients = torch.autograd.grad(actual.square().mean(), inputs)
        expected_gradients = torch.autograd.grad(expected.square().mean(), inputs)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient)


if __name__ == "__main__":
    unittest.main()
