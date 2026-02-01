import torch

tensor0d = torch.tensor(1)
tensor1d = torch.tensor([1, 2, 3])
tensor2d = torch.tensor([[1, 2, 3],
                        [4, 5, 6]])
tensor3d = torch.tensor([[[1, 2], [3, 4]],
                        [[5, 6], [7, 8]]])

print(tensor2d.dtype)
print(tensor2d)
print(tensor2d.shape)
print(tensor2d.T)
print(tensor2d.matmul(tensor2d.T))