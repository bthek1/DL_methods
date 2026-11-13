# Save and Load

Save and Load

Author

Benedict Thekkel

``` python
import torch
import torch.nn as nn
```

## Save arg dict with python pickle

``` python
torch.save(arg, PATH)
torch.load(PATH)
model.load_state_dict(arg)
```

## Save model with python pickle

``` python
torch.save(model, PATH)

model = torch.load(PATH)

model.eval()
```

## Recommented Method

``` python
torch.save(model.state_dict(), PATH)

model = Model(*args, **kwargs)

model.load_state_dict(torch.load(PATH))

model.eval()
```

## Test model

``` python
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
```

``` python
device = torch.device("cuda")
model = Model(n_input_features = 6)
model
```

    Model(
      (linear): Linear(in_features=6, out_features=1, bias=True)
    )

## Method 1

``` python
FILE = "model.pth1"
torch.save(model, FILE)
```

``` python
model = torch.load(FILE)
model.eval()
```

    Model(
      (linear): Linear(in_features=6, out_features=1, bias=True)
    )

``` python
for param in model.parameters():
    print(param)
```

    Parameter containing:
    tensor([[-0.3002, -0.2477, -0.2695, -0.1810, -0.0604,  0.1516]],
           requires_grad=True)
    Parameter containing:
    tensor([-0.3233], requires_grad=True)

## Method 2

``` python
FILE = "model.pth2"
torch.save(model.state_dict(), FILE)
```

``` python
model = Model(n_input_features = 6)
```

``` python
model.load_state_dict(torch.load(FILE))

model.eval()
```

    Model(
      (linear): Linear(in_features=6, out_features=1, bias=True)
    )

``` python
for param in model.parameters():
    print(param)
```

    Parameter containing:
    tensor([[-0.3002, -0.2477, -0.2695, -0.1810, -0.0604,  0.1516]],
           requires_grad=True)
    Parameter containing:
    tensor([-0.3233], requires_grad=True)

``` python
model.state_dict()
```

    OrderedDict([('linear.weight',
                  tensor([[-0.3002, -0.2477, -0.2695, -0.1810, -0.0604,  0.1516]])),
                 ('linear.bias', tensor([-0.3233]))])

``` python
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
optimizer.state_dict()
```

    {'state': {},
     'param_groups': [{'lr': 0.01,
       'momentum': 0,
       'dampening': 0,
       'weight_decay': 0,
       'nesterov': False,
       'maximize': False,
       'foreach': None,
       'differentiable': False,
       'params': [0, 1]}]}

``` python
checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict()
}
checkpoint
```

    {'epoch': 90,
     'model_state': OrderedDict([('linear.weight',
                   tensor([[-0.3002, -0.2477, -0.2695, -0.1810, -0.0604,  0.1516]])),
                  ('linear.bias', tensor([-0.3233]))]),
     'optim_state': {'state': {},
      'param_groups': [{'lr': 0.01,
        'momentum': 0,
        'dampening': 0,
        'weight_decay': 0,
        'nesterov': False,
        'maximize': False,
        'foreach': None,
        'differentiable': False,
        'params': [0, 1]}]}}

``` python
torch.save(checkpoint, "checkpoint.pth")
```

``` python
loaded_checkpoint = torch.load("checkpoint.pth")
```

``` python
loaded_checkpoint['epoch']
```

    90

``` python
model.load_state_dict(loaded_checkpoint['model_state'])
model.state_dict()
```

    OrderedDict([('linear.weight',
                  tensor([[-0.3002, -0.2477, -0.2695, -0.1810, -0.0604,  0.1516]])),
                 ('linear.bias', tensor([-0.3233]))])

``` python
optimizer.load_state_dict(loaded_checkpoint['optim_state'])
optimizer.state_dict()
```

    {'state': {},
     'param_groups': [{'lr': 0.01,
       'momentum': 0,
       'dampening': 0,
       'weight_decay': 0,
       'nesterov': False,
       'maximize': False,
       'foreach': None,
       'differentiable': False,
       'params': [0, 1]}]}

Back to top
