import torch
from itertools import product as iproduct
import sys
import math
from pathlib import Path
_root = Path(__file__).parent.parent  # Setting root
sys.path.append(str(_root / "jointCode"))
sys.path.append(str(_root / "fockBackend"))
torch.set_default_dtype(torch.float64)

import strawberryfields as sf
from strawberryfields import ops

# Initialize device dynamically based on config
n_wires = CONFIG.circuit.wires
n_dims = CONFIG.circuit.dims
