```import torch 

def activation(x):
  return 1/(1+torch.exp(-x))

torch.manual_seed(7)

features=torch.randn((1,5))
weights = torch.randn_like(features)
bias=torch.randn((1,1))
# making Labels from our data and true weights  
y = activation(torch.sum(features * weights) + bias)
y = activation((features * weights).sum() + bias)
# will give error torch.mm(features, weights)
# since for the matrix multiplication both the matrices should be of compatible dimensions
# hence we can use reshape or view to resize the veiwes
y = activation(torch.mm(features, weights.view(5,1)) + bias)

print (y)

```