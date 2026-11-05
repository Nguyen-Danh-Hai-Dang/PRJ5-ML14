from torch import stack, tril
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
# import torch
from torch.nn import Parameter, Linear
from torch import tensor, tensordot, ones, matmul, zeros 
from torch.nn.functional import relu, softmax
from torch import movedim
import torch
import torch.nn as nn

"""
##################
### QUESTION 1 ###
##################
"""


class PerceptronModel(Module):
    def __init__(self, dimensions):
        super(PerceptronModel, self).__init__()
        self.w = nn.Parameter(torch.zeros(1, dimensions))

    def get_weights(self):
        return self.w

    def forward(self, x):
        return torch.sum(x * self.w, dim=1)

    def get_prediction(self, x):
        score = self(x)
        if score.item() >= 0:
            return 1
        else:
            return -1

class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(1, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)

        return x


class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        # Two hidden layers learn nonlinear features from flattened pixels.
        self.layer1 = Linear(input_size, 256)
        self.layer2 = Linear(256, 128)
        self.output = Linear(128, output_size)


    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        # Cross entropy in the next step will consume raw logits.
        return self.output(x)



class LanguageIDModel(Module):

    def __init__(self):
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()        

        self.hidden_size = 100
        
        self.input_layer = Linear(self.num_chars, self.hidden_size)
        self.hidden_layer = Linear(self.hidden_size, self.hidden_size)
        self.output_layer = Linear(self.hidden_size, len(self.languages))
            
    def forward(self, xs):
        batch_size = xs[0].shape[0]
               
        hidden = xs[0].new_zeros(
            (batch_size, self.hidden_size)
        )

        for x in xs:
            hidden = relu(
                self.input_layer(x) +
                self.hidden_layer(hidden)
            )

        logits = self.output_layer(hidden)

        return logits



def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'

    The method 'zeros((y_dim,x_dim))' may also be useful. It initializes a pytorch tensor with dimensions (y_dim, x_dim), with every value
    set to zero.
    """
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    Output_Tensor = tensor(())
    "*** YOUR CODE HERE ***"

    "*** End Code ***"
    return Output_Tensor


class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.

    Note that this class looks different from a standard pytorch model since we don't need to train it
    as it will be run on preset weights.
    """

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))
        """ YOUR CODE HERE """


    def forward(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimensional datapoint now, similar to the previous questions.
        """
        x = x.reshape(len(x), 28, 28)
        x = stack(
            list(map(lambda sample: Convolve(sample, self.convolution_weights), x))
        )
        x = x.flatten(start_dim=1)
        """ YOUR CODE HERE """


class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        """
        All the layers you should use are defined here.

        In order to pass the autograder, make sure each linear layer matches up with their corresponding matrix,
        ie: use self.k_layer to generate the K matrix.
        """
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size, layer_size)

        # Masking part of attention layer
        self.register_buffer(
            "mask",
            tril(ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

        self.layer_size = layer_size

    def forward(self, input):
        """
        Applies the attention mechanism to input. All necessary layers have
        been defined in __init__()

        In order to apply the causal mask to a given matrix M, you should update
        it as such:

        M = M.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]

        For the softmax activation, it should be applied to the last dimension of the input,
        Take a look at the "dim" argument of torch.nn.functional.softmax to figure out how to do this.
        """
        B, T, C = input.size()

        Q = self.q_layer(input)
        K = self.k_layer(input)
        V = self.v_layer(input)

        scores = matmul(Q, movedim(K, 1, 2)) / (self.layer_size ** 0.5)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))[0]
        weights = softmax(scores, dim=-1)

        return matmul(weights, V)

    