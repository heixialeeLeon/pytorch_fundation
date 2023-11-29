import torch
from tensor.tensor_base import show_tensor_info

"""
作用：与 view 方法类似，将输入 Tensor 转换为新的 shape 格式。

但是 reshape 方法更强大，可以认为：a.reshape = a.view() + a.contiguous().view()

即：在满足 Tensor 连续性条件时，a.reshape 返回的结果与 a.view() 相同，否则返回的结果与a.contiguous().view() 相同。
"""

data = torch.randn(4, 120, 8)
show_tensor_info((data))

print("使用reshape进行transpose ********")
data_1 = data.reshape(4,8,120)
show_tensor_info(data_1)

print("使用reshape进行升维 ********")
data_2 = data.reshape(4,4,30,8)
show_tensor_info(data_2)

print("使用reshape进行降维 ********")
data_3 = data.reshape(4,960)
show_tensor_info(data_3)