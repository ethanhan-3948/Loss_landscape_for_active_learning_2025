""" Class used to define interface to complex models """

import abc
import itertools
import torch.nn
from .model_parameters import ModelParameters


class ModelWrapper(abc.ABC):
    def __init__(self, modules: list):
        self.modules = modules

    def get_modules(self) -> list:
        return self.modules

    def get_torch_module_parameters(self) -> ModelParameters:
        param_list = []
        for layer_name in self.modules[0].state_dict().keys():
            if layer_name.split('.')[-1] == 'num_batches_tracked':
                continue
            
            p = self.modules[0].state_dict()[layer_name].float()
            if len(p.shape) == 0:
                p = p.expand(1)

            param_list.append(p)
        return ModelParameters(param_list)

    def get_module_parameters(self) -> ModelParameters:
        return ModelParameters([p for module in self.modules for p in module.parameters()])

    def train(self, mode=True) -> 'ModelWrapper':
        for module in self.modules:
            module.train(mode)
        return self

    def eval(self) -> 'ModelWrapper':
        return self.train(False)

    def requires_grad_(self, requires_grad=True) -> 'ModelWrapper':
        for module in self.modules:
            for p in module.parameters():
                p.requires_grad = requires_grad
        return self

    def zero_grad(self) -> 'ModelWrapper':
        for module in self.modules:
            for p in module.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
        return self

    def parameters(self):
        return itertools.chain([module.parameters() for module in self.modules])

    def named_parameters(self):
        return itertools.chain([module.named_parameters() for module in self.modules])

    @abc.abstractmethod
    def forward(self, x):
        pass


class SimpleModelWrapper(ModelWrapper):
    def __init__(self, model: torch.nn.Module):
        super().__init__([model])

    def forward(self, x):
        return self.modules[0](x)


class GeneralModelWrapper(ModelWrapper):
    def __init__(self, model, modules: list, forward_fn):
        super().__init__(modules)
        self.model = model
        self.forward_fn = forward_fn

    def forward(self, x):
        return self.forward_fn(self.model, x)


def wrap_model(model):
    if isinstance(model, ModelWrapper):
        return model.requires_grad_(False)
    elif isinstance(model, torch.nn.Module):
        return SimpleModelWrapper(model).requires_grad_(False)
    else:
        raise ValueError('Only models of type torch.nn.modules.module.Module can be passed without a wrapper.')
