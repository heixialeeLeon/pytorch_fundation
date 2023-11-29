import torch
from tensor.tensor_base import show_tensor_info

data = torch.arange(8).reshape(2, 4)
show_tensor_info(data)

print("交换维度后，地址相同，没有发生copy，变为不连续了， 这个和transpose相似：******************")
data_1 = data.permute(1,0)
show_tensor_info(data_1)

print("********** 三维数据 ************")
data = torch.arange(16).reshape(2,2,4)
show_tensor_info(data)

print("三维数据变换 *****************")
data_1 = data.permute(2,1,0)
show_tensor_info(data_1)
