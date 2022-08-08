import torch
from torch import nn

def tversky(y_true, y_pred, beta, batch_mean=True):
    tp = torch.sum(y_true*y_pred, axis=(1,2,3))
    fn = torch.sum((1. - y_pred)*y_true, axis=(1,2,3))
    fp = torch.sum(y_pred*(1. - y_true), axis=(1,2,3))
    #tversky = reduce_mean(tp)/(reduce_mean(tp) + 
    #                           beta*reduce_mean(fn) + 
    #                           (1. - beta)*reduce_mean(fp))
    tversky = tp/\
        (tp + beta*fn + (1. - beta)*fp)
    
    norm = torch.sum(y_true, axis=(1,2,3)) + torch.sum(y_pred, axis=(1,2,3)) - tp
    if batch_mean:
        return torch.mean((1. - tversky)*norm)
    else:
        return (1. - tversky)*norm

adversarial_loss = nn.BCELoss() 

def generator_loss(generated_img, target_img):
    gen_loss = tversky(target_img, generated_img, beta = 0.7)
    return gen_loss

def discriminator_loss(output, label):
    disc_loss = 350*adversarial_loss(output, label)
    return disc_loss

