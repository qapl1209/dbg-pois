import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

def switch(condition, then_expression, else_expression):
    return torch.where(condition.bool(), then_expression, else_expression)

def lrelu(x, leak=0.2):
    return torch.where(x >= 0, x, leak * x)

def huber_loss(x, delta=1.0):
    return torch.where(
        torch.abs(x) < delta,
        0.5 * x ** 2,
        delta * (torch.abs(x) - 0.5 * delta)
    )

class GlobalSession:
    def __init__(self, num_cpu=None):
        self.num_cpu = num_cpu or torch.get_num_threads()
        
    def single_threaded_session(self):
        return GlobalSession(num_cpu=1)

def initialize():
    pass

def normc_initializer(std=1.0):
    def _initializer(shape):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return torch.tensor(out)
    return _initializer

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="same"):
    conv_layer = nn.Conv2d(in_channels=x.shape[1], out_channels=num_filters, kernel_size=filter_size, stride=stride, padding=pad)
    return conv_layer(x)

def function(inputs, outputs):
    def func(*args, **kwargs):
        input_dict = dict(zip(inputs, args))
        return outputs(**input_dict)
    return func

def var_shape(x):
    return list(x.size())

def numel(x):
    return x.numel()

def flatgrad(loss, var_list, clip_norm=None):
    grads = [torch.autograd.grad(loss, var, retain_graph=True)[0] for var in var_list]
    if clip_norm is not None:
        grads = [torch.nn.utils.clip_grad_norm_(grad, clip_norm) for grad in grads]
    return torch.cat([grad.view(-1) for grad in grads])

def assignFromFlat(var_list, values):
    start = 0
    for var in var_list:
        size = numel(var)
        var.data.copy_(values[start:start + size].view(var.shape))
        start += size

class SetFromFlat:
    def __init__(self, var_list):
        self.var_list = var_list
        self.total_size = sum(numel(var) for var in var_list)
        self.theta = torch.empty(self.total_size)

    def __call__(self, theta):
        assignFromFlat(self.var_list, theta)

class GetFlat:
    def __init__(self, var_list):
        self.var_list = var_list

    def __call__(self):
        return torch.cat([var.view(-1) for var in self.var_list])

def flattenallbut0(x):
    return x.view(x.size(0), -1)

def reduce_var(x, axis=None, keepdims=False):
    mean = torch.mean(x, dim=axis, keepdim=True)
    devs_squared = (x - mean) ** 2
    return torch.mean(devs_squared, dim=axis, keepdim=keepdims)

def reduce_std(x, axis=None, keepdims=False):
    return torch.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))
