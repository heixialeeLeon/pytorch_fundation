import torch

"""
@ 和 * 代表矩阵的两种相乘方式：
@ 表示常规的数学上定义的矩阵相乘；
* 表示两个矩阵对应位置处的两个元素相乘
"""

x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[2, 1], [4, 3]])
c0 = x @ y
print("@ 操作结果")
print(c0)

print("* 操作结果")
c1 = x * y
print(c1)

print("torch.mm 操作结果，与符号@相同")
c2 = torch.mm(x,y)
print(c2)

print("torch.mul 操作结果，与符号*相同")
c3 = torch.mul(x,y)
print(c3)