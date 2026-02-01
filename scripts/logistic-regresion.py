# Prevent long lines of code
import torch
import torch.nn.functional as F
from torch.autograd import grad

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
# Bias unit
b = torch.tensor([0.0], requires_grad=True)
# Net input
z = x1 * w1 + b
# Activation and outpput
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)

# Save the graph un memory
grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

print(grad_L_w1)
print(grad_L_b)