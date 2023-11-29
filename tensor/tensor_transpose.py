import torch
from tensor.tensor_base import show_tensor_info

data = torch.arange(8).reshape(2, 4)
show_tensor_info(data)

print("交换维度后，地址相同，没有发生copy，变为不连续了：******************")
data_1 = data.transpose(0,1)
show_tensor_info(data_1)

print("变化维度 + contignous， 发生了copy，地址改变了： ****************")
data_2 = data.transpose(0,1).contiguous()
show_tensor_info(data_2)
