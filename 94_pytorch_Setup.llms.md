# OLD: Pytorch Setup

Details on setting up Pytorch Setup

Author

Benedict Thekkel

Code

``` python
import torch
```

``` python
x = torch.rand(5,3)
print(x)
```

    tensor([[0.5191, 0.8232, 0.7263],
            [0.0213, 0.3196, 0.3647],
            [0.9279, 0.8826, 0.0079],
            [0.5056, 0.1073, 0.1252],
            [0.4052, 0.7890, 0.1353]])

``` python
torch.cuda.is_available()
```

    True

``` python
torch.cuda.device_count()
```

    1

``` python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
```

    device(type='cuda')

``` python
!nvcc --version
```

    fish: Unknown command: nvcc
    fish: 
    nvcc --version
    ^~~^

``` python
!nvidia-smi
```

    Fri Jun 19 13:44:12 2026       
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 595.71.05              Driver Version: 595.71.05      CUDA Version: 13.2     |
    +-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA GeForce GTX 1660 ...    Off |   00000000:01:00.0  On |                  N/A |
    | 50%   48C    P5             15W /  125W |    3259MiB /   6144MiB |     21%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+

    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |    0   N/A  N/A            3912      G   /usr/bin/gnome-shell                   1541MiB |
    |    0   N/A  N/A            4442      G   ...exec/xdg-desktop-portal-gnome          2MiB |
    |    0   N/A  N/A            4870      G   /usr/lib/firefox/firefox               1512MiB |
    |    0   N/A  N/A            6980      G   /usr/bin/Xwayland                         2MiB |
    +-----------------------------------------------------------------------------------------+

``` python
!which python
```

    /home/proxmox-ml5/.platformio/penv/bin/python

``` python
import sys
```

``` python
print(sys.executable)
print(sys.version)
```

    /home/proxmox-ml5/Documents/Knowledge/.venv/bin/python3
    3.14.0b3 (main, Jun 29 2025, 16:27:06) [Clang 20.1.4 ]

Back to top
