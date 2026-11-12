from torch.nn.functional import mse_loss, cross_entropy
import torch.nn.functional as F

def regression_loss(y_pred, y):
    loss = mse_loss(y_pred, y)
    return loss.reshape(1)



def digitclassifier_loss(y_pred, y):
    return cross_entropy(y_pred, y).reshape(1)


def languageid_loss(predicted_logits, labels):
    return F.cross_entropy(predicted_logits, labels)


def digitconvolution_Loss(y_pred, y):
    return cross_entropy(y_pred, y).reshape(1)
    
