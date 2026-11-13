import argparse
import strawberryfields as sf
from strawberryfields.ops import Dgate, Sgate, Rgate, BSgate
import tensorflow as tf

# ==============================================================================
# 1. TOPOLOGY HELPER
# ==============================================================================

def get_bs_pairs(n_wires, topology):
    """Returns a list of (wire1, wire2) tuples based on the chosen architecture."""
    if topology == "ladder":
        # Wires only connect to their immediate neighbors (0-1, 1-2)
        return [(i, i + 1) for i in range(n_wires - 1)]
    elif topology == "ring":
        # Ladder, but the last wire loops back to the first (0-1, 1-2, 2-0)
        return [(i, (i + 1) % n_wires) for i in range(n_wires)]
    elif topology == "all-to-all":
        # Every wire connects to every other wire
        pairs = []
        for i in range(n_wires):
            for j in range(i + 1, n_wires):
                pairs.append((i, j))
        return pairs
    else:
        raise ValueError(f"Unknown topology: {topology}")

# ==============================================================================
# 2. ARGUMENTS & WEIGHT INITIALIZATION
# ==============================================================================

def parse_terminal_args():
    """Catches inputs from the HPC terminal job script."""
    parser = argparse.ArgumentParser(description="QML Particle Physics Circuit")
    
    parser.add_argument("--n_wires", type=int, default=3)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--circ_seq", type=str, default="br", 
                        help="Sequence of variational gates (e.g., 'br', 'bsr')")
    parser.add_argument("--topology", type=str, default="ladder", 
                        choices=["ladder", "ring", "all-to-all"],
                        help="Entanglement architecture for Beamsplitters")
    parser.add_argument("--cutoff_dim", type=int, default=4,
                        help="Fock space cutoff dimension for the simulation")
    
    args = parser.parse_args()
    return vars(args)

def initialize_weights(config):
    """Allocates TensorFlow variables dynamically based on topology and sequence."""
    params = {}
    l = config["num_layers"]
    w = config["n_wires"]
    
    # Calculate exact number of beamsplitters needed for this topology
    bs_pairs = get_bs_pairs(w, config["topology"])
    num_bs = len(bs_pairs)
    
    shape_single = (l, w)
    shape_pairs = (l, num_bs) 

    # --- Static Encoding Weights (Displacement & Squeezing) ---
    params["enc_d_pt"] = tf.Variable(tf.random.normal(shape_single, stddev=0.1))
    params["enc_d_eta"] = tf.Variable(tf.random.normal(shape_single, stddev=0.1))
    params["enc_s_mag"] = tf.Variable(tf.random.normal(shape_single, stddev=0.1))
    params["enc_s_phi"] = tf.Variable(tf.random.normal(shape_single, stddev=0.1))

    # --- Dynamic Variational Weights ---
    circ_seq = config["circ_seq"].lower()
    
    if 'b' in circ_seq:
        params["var_b_theta"] = tf.Variable(tf.random.normal(shape_pairs, stddev=0.1))
        params["var_b_phi"] = tf.Variable(tf.random.normal(shape_pairs, stddev=0.1))
    if 'r' in circ_seq:
        params["var_r_angle"] = tf.Variable(tf.random.normal(shape_single, stddev=0.1))
    if 's' in circ_seq:
        params["var_s_mag"] = tf.Variable(tf.random.normal(shape_single, stddev=0.1))
        params["var_s_phi"] = tf.Variable(tf.random.normal(shape_single, stddev=0.1))
    if 'd' in circ_seq:
        params["var_d_mag"] = tf.Variable(tf.random.normal(shape_single, stddev=0.1))
        params["var_d_phi"] = tf.Variable(tf.random.normal(shape_single, stddev=0.1))
        
    return params

# ==============================================================================
# 3. QUANTUM CIRCUIT DEFINITIONS
# ==============================================================================

def static_polar_encoding(x, n_wires, params, layer=0):
    """Statically embeds [eta, pt, phi] using Displacement then Squeezing."""
    for wire in range(n_wires):
        eta, pt, phi = x[wire, 0], x[wire, 1], x[wire, 2]

        w_pt = params["enc_d_pt"][layer, wire]
        w_eta = params["enc_d_eta"][layer, wire]
        w_sq_r = params["enc_s_mag"][layer, wire]
        w_phi = params["enc_s_phi"][layer, wire]

        Dgate(w_pt * pt, w_eta * eta) | wire
        Sgate(w_sq_r, w_phi * phi) | wire

def dynamic_variational_layer(n_wires, params, circ_seq, topology, layer=0):
    """Constructs the trainable layer using the requested gate sequence and topology."""
    for gate_char in circ_seq.lower():
        
        if gate_char == 'b':
            # Fetch the specific wiring pairs for this architecture
            bs_pairs = get_bs_pairs(n_wires, topology)
            
            for bs_idx, (wire1, wire2) in enumerate(bs_pairs):
                w_theta = params["var_b_theta"][layer, bs_idx]
                w_phi = params["var_b_phi"][layer, bs_idx]
                BSgate(w_theta, w_phi) | (wire1, wire2)
                
        elif gate_char == 'r':
            for i in range(n_wires):
                Rgate(params["var_r_angle"][layer, i]) | i
                
        elif gate_char == 's':
            for i in range(n_wires):
                Sgate(params["var_s_mag"][layer, i], params["var_s_phi"][layer, i]) | i
                
        elif gate_char == 'd':
            for i in range(n_wires):
                Dgate(params["var_d_mag"][layer, i], params["var_d_phi"][layer, i]) | i
                
        else:
            raise ValueError(f"Unsupported variational gate: '{gate_char}'")

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # 1. Catch HPC Configurations
    config = parse_terminal_args()
    print(f"--- Building QML Circuit with Config: {config} ---")
    
    # 2. Allocate variables precisely
    trainable_params = initialize_weights(config)
    
    # 3. Create dummy batch of kinematic data (e.g., 1 event with `n_wires` particles)
    x_data = tf.random.uniform((config["n_wires"], 3), minval=0.1, maxval=2.0)
    
    # 4. Construct DAG Blueprint
    prog = sf.Program(config["n_wires"])
    with prog.context:
        for layer in range(config["num_layers"]):
            static_polar_encoding(x_data, config["n_wires"], trainable_params, layer)
            
            # Now passing `topology` down to the variational layer
            dynamic_variational_layer(
                config["n_wires"], trainable_params, 
                config["circ_seq"], config["topology"], layer
            )

    # 5. Run on TensorFlow Backend
    eng = sf.Engine("tf", backend_options={"cutoff_dim": cutoff_dim})
    result = eng.run(prog)
    
    print("Execution complete. Final Quantum State tensor shape:", result.state.ket().shape)