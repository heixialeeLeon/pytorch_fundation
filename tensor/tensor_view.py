import torch
from tensor.tensor_base import show_tensor_info

data = torch.randn(4, 120, 8)
show_tensor_info((data))

print("使用view进行transpose ********")
data_1 = data.view(4,8,120)
show_tensor_info(data_1)

print("使用view进行升维 ********")
data_2 = data.view(4,4,30,8)
show_tensor_info(data_2)

print("使用view进行降维 ********")
data_3 = data.view(4,960)
show_tensor_info(data_3)