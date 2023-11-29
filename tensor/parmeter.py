"""
Parameter is sub-classed from Tensor it is a Tensor.
But there is a trick. Parameters that are inside of a module are added to the list of Module parameters.
If m is your module m.parameters() will hold your parameter.

a nn.Module class, doesn't always explicitly knows what Tensor objects it should optimize for.
If you go through this simple commented piece of code, it could clarify it further.
"""

import torch
import torch.nn as nn

# Model 1
class M1(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(2, 2))
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        return x @ self.weights + self.bias

# Model 2
class M2(nn.Module):
    def __init__(self):
        super().__init__()

        # though the Tensor Objects below can undergo backprop and minimize some loss
        # our model class doesn't know, it should use these tensors during optimization
        self.weights = torch.randn(2,2).requires_grad_(True)
        self.bias = torch.zeros(2).requires_grad_(True)

    def forward(self, x):
        return x @ self.weights + self.bias

m1 = M1()
m2 = M2()
print("the M1 model with Parameters")
print(list(m1.parameters()))

print("The M2 model with Tensor")
print(list(m2.parameters()))