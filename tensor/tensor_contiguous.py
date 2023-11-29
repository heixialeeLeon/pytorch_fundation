import torch
from tensor.tensor_base import show_tensor_info

data = torch.arange(9).reshape(3, 3)
show_tensor_info(data)

data_1 = data.permute(1,0)
show_tensor_info(data_1)

# this will trigger error
# data_2 = data_1.view(9)
print("使用contiguous后，重新分配内存，导致地址与data_1不一样")
data_2 = data_1.contiguous()
show_tensor_info(data_2)