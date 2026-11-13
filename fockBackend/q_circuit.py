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

import strawberryfields as sf
from strawberryfields.ops import Dgate, Sgate

def sf_polar_encoding(x, n_wires, params, q, reupload_layer=0):
    """
    Encodes classical kinematic variables (Pt, eta, phi) directly into the polar 
    parameters of continuous-variable quantum gates.

    Parameters:
    -----------
    x : tf.Tensor or numpy.ndarray
        A 2D array of classical features with shape (n_wires, 3). 
        The second dimension is expected to be [eta, pt, phi].
    n_wires : int
        The number of qumodes (wires) in the quantum circuit.
    params : dict
        A dictionary containing the trainable weights. Expected keys are 
        "enc_var_pt", "enc_var_eta", "enc_var_sq_r", and "enc_var_phi". 
        Each key should map to a 2D tensor of shape (num_layers, n_wires).
    reupload_layer : int, optional (default=0)
        The current layer index in the data re-uploading architecture.

    ========================================================================
    DISCLAIMER: Zero-Value Behavior in Polar Phase Space
    ========================================================================
    Because this function maps features directly to polar coordinates (radius and 
    angle), you must be aware of how 0-values behave geometrically:

    1. The eta = 0 Behavior:
       If `eta = 0` (or `w_eta * eta = 0`), the phase angle of the Displacement 
       gate is exactly 0. The state is displaced entirely along the X-axis 
       (the position quadrature). It receives no rotational shift.

    2. The pt = 0 (Vanishing Feature) Risk:
       If `pt = 0` (or `w_pt * pt = 0`), the radius of the Displacement is 0, 
       meaning the quantum state remains dead-center at the vacuum origin. 
       Because rotating a state at the origin does nothing, the `eta` phase 
       parameter is completely ignored. If your dataset contains true 0 values 
       for Pt, the network will be temporarily blind to eta for that particle.
    ========================================================================
    """
    for wire in range(n_wires):
        eta = x[wire, 0]
        pt = x[wire, 1]
        phi = x[wire, 2]

        w_pt = params["enc_var_pt"][reupload_layer, wire]
        w_eta = params["enc_var_eta"][reupload_layer, wire]
        w_sq_r = params["enc_var_sq_r"][reupload_layer, wire] 
        w_phi = params["enc_var_phi"][reupload_layer, wire]

        Dgate(w_pt * pt, w_eta * eta) | wire
        Sgate(w_sq_r, w_phi * phi) | wire