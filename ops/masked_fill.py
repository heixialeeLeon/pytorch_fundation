import torch

"""
masked_fill方法有两个参数，maske和value，mask是一个pytorch张量（Tensor），元素是布尔值，
value是要填充的值，填充规则是mask中取值为True位置对应于self的相应位置用value填充。
"""

data = torch.randn(3,2)
print("data is:")
print(data)

mask = torch.randint(0,2,(3,2))
mask = mask ==0
print("mask is")
print(mask)

result = data.masked_fill(mask, -1000)
print("result is ")
print(result)