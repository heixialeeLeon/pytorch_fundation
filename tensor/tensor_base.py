import torch

data = torch.randn(4, 120, 8)

"""
stride[i] = stride[i+1]*size[i+1]
"""

def show_tensor_info(data):
    print("address: {}".format(hex(data.storage().data_ptr())))
    print("dim: {}".format(data.dim()))
    print("size: {}".format(data.size()))
    print("stride: {}".format(data.stride()))
    print("offset: {}".format(data.storage_offset()))

if __name__ == "__main__":
    show_tensor_info(data)