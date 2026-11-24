import argparse
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for HPC nodes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from types import SimpleNamespace
from data_loader import TTBarZJetsData
import strawberryfields as sf
from strawberryfields.ops import Dgate, Sgate, Rgate, BSgate
import tensorflow as tf
tf.get_logger().setLevel("ERROR")


# ==============================================================================
# 1. ARGUMENT PARSING & TOPOLOGY HELPER
# ==============================================================================

def parse_terminal_args():
    """Catches hyperparameters and circuit architecture flags from the HPC terminal."""
    parser = argparse.ArgumentParser(description="CV-QML Particle Physics Training & Evaluation")

    # Circuit Architecture
    parser.add_argument("--n_wires", type=int, default=3,
                        help="Number of qumodes/particles per event")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of data re-uploading layers")
    parser.add_argument("--circ_seq", type=str, default="br",
                        help="Variational gate sequence (e.g., 'br', 'bsr', 'dbr')")
    parser.add_argument("--topology", type=str, default="ladder",
                        choices=["ladder", "ring", "all-to-all"],
                        help="Beamsplitter entanglement topology")
    parser.add_argument("--readout", type=str, default="all",
                        choices=["single", "all"],
                        help="Homodyne readout: wire 0 only ('single') or all wires + Dense ('all')")

    # Training & Simulator Hyperparameters
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01, help="Adam learning rate")
    parser.add_argument("--cutoff_dim", type=int, default=4,
                        help="Fock space photon cutoff dimension")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to save metrics, weights, and ROC plots")
    
    # Dataset & Qumode Scaling Hyperparameters
    parser.add_argument("--dataset_path", type=str,
                        default="/rds/general/user/av523/home/projects/myproject/data/QMLoutputs/TTBar_+ZJetsToNuNU_flat_merged.h5",
                        help="Path to the merged HDF5 jet dataset")
    parser.add_argument("--train_samples", type=int, default=2000,
                        help="Total supervised training jets (split 50/50 sig/bkg)")
    parser.add_argument("--val_samples", type=int, default=1000,
                        help="Total validation jets (split 50/50 sig/bkg)")
    parser.add_argument("--test_samples", type=int, default=1000,
                        help="Total test jets (split 50/50 sig/bkg)")
    parser.add_argument("--d_mult", type=float, default=1.0,
                        help="Displacement scaling multiplier for pt_frac")
    parser.add_argument("--s_mult", type=float, default=1.0,
                        help="Squeezing scaling multiplier for eta")

    return vars(parser.parse_args())


def get_bs_pairs(n_wires, topology):
    """Returns a list of (wire1, wire2) tuples based on the chosen topology."""
    if topology == "ladder":
        return [(i, i + 1) for i in range(n_wires - 1)]
    elif topology == "ring":
        return [(i, (i + 1) % n_wires) for i in range(n_wires)]
    elif topology == "all-to-all":
        return [(i, j) for i in range(n_wires) for j in range(i + 1, n_wires)]
    else:
        raise ValueError(f"Unknown topology: {topology}")


# ==============================================================================
# 2. DATA LOADING (REPLACE WITH YOUR ACTUAL PHYSICS DATASET)
# ==============================================================================

def load_dataset(config):
    """Adapts flat CLI config into the nested cfg expected by TTBarZJetsData."""
    cfg = SimpleNamespace(
        circuit=SimpleNamespace(wires=config["n_wires"]),
        data=SimpleNamespace(
            dataset_path=config["dataset_path"],
            train_num_samples=config["train_samples"],
            val_num_samples=config["val_samples"],
            test_num_samples=config["test_samples"],
            d_mult=config["d_mult"],
            s_mult=config["s_mult"],
        ),
    )
    rng = np.random.default_rng(config["seed"])
    data_handler = TTBarZJetsData(cfg)
    x_train, y_train, x_val, y_val, x_test, y_test = data_handler.createSupervisedDatasets(rng)
    data_handler.free_raw_data()
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    return (x_all[:n_train], y_all[:n_train]), (x_all[n_train:], y_all[n_train:])


# ==============================================================================
# 3. MODEL & WEIGHT INITIALIZATION
# ==============================================================================

def initialize_model(config):
    """Allocates quantum gate variables and the optional classical readout layer."""
    params = {}
    l = config["num_layers"]
    w = config["n_wires"]

    bs_pairs = get_bs_pairs(w, config["topology"])
    shape_single = (l, w)
    shape_pairs = (l, len(bs_pairs))

    # Static Polar Encoding Weights
    params["enc_d_pt"] = tf.Variable(tf.random.normal(shape_single, stddev=0.1), name="enc_d_pt")
    params["enc_d_eta"] = tf.Variable(tf.random.normal(shape_single, stddev=0.1), name="enc_d_eta")
    params["enc_s_mag"] = tf.Variable(tf.random.normal(shape_single, stddev=0.1), name="enc_s_mag")
    params["enc_s_phi"] = tf.Variable(tf.random.normal(shape_single, stddev=0.1), name="enc_s_phi")

    # Dynamic Variational Weights
    circ_seq = config["circ_seq"].lower()
    if 'b' in circ_seq:
        params["var_b_theta"] = tf.Variable(tf.random.normal(shape_pairs, stddev=0.1), name="var_b_theta")
        params["var_b_phi"] = tf.Variable(tf.random.normal(shape_pairs, stddev=0.1), name="var_b_phi")
    if 'r' in circ_seq:
        params["var_r_angle"] = tf.Variable(tf.random.normal(shape_single, stddev=0.1), name="var_r_angle")
    if 's' in circ_seq:
        params["var_s_mag"] = tf.Variable(tf.random.normal(shape_single, stddev=0.1), name="var_s_mag")
        params["var_s_phi"] = tf.Variable(tf.random.normal(shape_single, stddev=0.1), name="var_s_phi")
    if 'd' in circ_seq:
        params["var_d_mag"] = tf.Variable(tf.random.normal(shape_single, stddev=0.1), name="var_d_mag")
        params["var_d_phi"] = tf.Variable(tf.random.normal(shape_single, stddev=0.1), name="var_d_phi")

    readout_layer = None
    if config["readout"] == "all":
        readout_layer = tf.keras.layers.Dense(1, name="classical_readout")
        readout_layer.build(input_shape=(None, w))

    return params, readout_layer


# ==============================================================================
# 4. QUANTUM CIRCUIT & HOMODYNE READOUT
# ==============================================================================

def static_polar_encoding(x_batch, n_wires, params, layer=0):
    """Embeds [eta, pt, phi] across a batch using Displacement then Squeezing."""
    for wire in range(n_wires):
        eta = x_batch[:, wire, 0]
        pt  = x_batch[:, wire, 1]
        phi = x_batch[:, wire, 2]

        Dgate(params["enc_d_pt"][layer, wire] * pt,
              params["enc_d_eta"][layer, wire] * eta) | wire
        Sgate(params["enc_s_mag"][layer, wire],
              params["enc_s_phi"][layer, wire] * phi) | wire


def dynamic_variational_layer(n_wires, params, circ_seq, topology, layer=0):
    """Applies trainable gates in the sequence and Beamsplitter topology requested."""
    for gate_char in circ_seq.lower():
        if gate_char == 'b':
            for bs_idx, (w1, w2) in enumerate(get_bs_pairs(n_wires, topology)):
                BSgate(params["var_b_theta"][layer, bs_idx],
                       params["var_b_phi"][layer, bs_idx]) | (w1, w2)
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


def extract_homodyne_logits(state, config, readout_layer):
    """Performs fixed X-quadrature (phi=0.0) homodyne expectation readout."""
    if config["readout"] == "single":
        x_mean = state.quad_expectation(mode=0, phi=0.0)[0]
        return tf.expand_dims(x_mean, axis=-1)
    elif config["readout"] == "all":
        wire_means = [
            state.quad_expectation(mode=w, phi=0.0)[0]
            for w in range(config["n_wires"])
        ]
        quantum_features = tf.stack(wire_means, axis=1)
        return readout_layer(quantum_features)


def forward_pass(x_batch, config, params, readout_layer):
    """Builds and runs one batched forward pass of the quantum circuit."""
    current_bs = int(x_batch.shape[0])
    eng = sf.Engine(
        "tf",
        backend_options={
            "cutoff_dim": config["cutoff_dim"],
            "batch_size": current_bs
        }
    )

    prog = sf.Program(config["n_wires"])
    with prog.context:
        for layer in range(config["num_layers"]):
            static_polar_encoding(x_batch, config["n_wires"], params, layer)
            dynamic_variational_layer(
                config["n_wires"], params,
                config["circ_seq"], config["topology"], layer
            )

    result = eng.run(prog)
    return extract_homodyne_logits(result.state, config, readout_layer)


# ==============================================================================
# 5. TRAINING STEP, INFERENCE & EVALUATION METRICS
# ==============================================================================

def train_step(x_batch, y_batch, config, params, readout_layer, optimizer, loss_fn):
    """Executes one gradient descent update step on a batch."""
    with tf.GradientTape() as tape:
        logits = forward_pass(x_batch, config, params, readout_layer)
        loss = loss_fn(y_batch, logits)

    trainable_vars = list(params.values())
    if readout_layer is not None:
        trainable_vars += readout_layer.trainable_variables

    gradients = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))
    return loss


def predict_logits(x_data, config, params, readout_layer):
    """Runs pure forward inference across a dataset without gradient overhead."""
    num_samples = x_data.shape[0]
    batch_size = config["batch_size"]
    all_logits = []

    for start_idx in range(0, num_samples, batch_size):
        x_batch = x_data[start_idx : start_idx + batch_size]
        logits = forward_pass(x_batch, config, params, readout_layer)
        all_logits.append(logits.numpy())

    return np.concatenate(all_logits, axis=0)


import csv
import json
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def evaluate_and_save_results(x_val, y_val, history, config, params, readout_layer):
    """Computes final metrics and saves plottable CSVs, weights, and config in a run-specific folder."""
    run_id = (
        f"w{config['n_wires']}_l{config['num_layers']}_"
        f"{config['circ_seq']}_{config['topology']}_{config['readout']}"
    )
    
    # Create a dedicated directory for this specific run
    run_dir = os.path.join(config["save_dir"], run_id)
    os.makedirs(run_dir, exist_ok=True)

    # 1. Inference & Probabilities
    logits = predict_logits(x_val, config, params, readout_layer)
    probs = tf.sigmoid(logits).numpy().ravel()
    y_true = np.asarray(y_val).ravel()

    # 2. Metrics
    auc_score = roc_auc_score(y_true, probs)
    acc = accuracy_score(y_true, (probs >= 0.5).astype(int))
    fpr, tpr, thresholds = roc_curve(y_true, probs)

    idx_50 = np.argmin(np.abs(tpr - 0.5))
    bg_rejection_50 = 1.0 / max(fpr[idx_50], 1e-6)

    # 3. Save Trained Weights
    weight_dict = {k: v.numpy() for k, v in params.items()}
    if readout_layer is not None:
        w_dense, b_dense = readout_layer.get_weights()
        weight_dict["dense_kernel"] = w_dense
        weight_dict["dense_bias"] = b_dense
    np.savez(os.path.join(run_dir, "weights.npz"), **weight_dict)

    # 4. Save Summary Metrics & Config (JSON)
    summary = {
        "run_id": run_id,
        "roc_auc": float(auc_score),
        "accuracy": float(acc),
        "bg_rejection_at_50pct_eff": float(bg_rejection_50),
        "config": config
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # 5. Save Training History (Losses & Epoch Metrics) as CSV
    if history and isinstance(history, dict):
        history_path = os.path.join(run_dir, "training_history.csv")
        keys = list(history.keys())
        num_epochs = len(history[keys[0]])
        
        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch"] + keys)
            for epoch in range(num_epochs):
                writer.writerow([epoch + 1] + [history[k][epoch] for k in keys])

    # 6. Save ROC Curve Points as CSV
    roc_path = os.path.join(run_dir, "roc_curve.csv")
    with open(roc_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fpr", "tpr", "threshold"])
        for fp, tp, thresh in zip(fpr, tpr, thresholds):
            writer.writerow([fp, tp, thresh])

    print("\n==================================================")
    print(f"FINAL EVALUATION [{run_id}]")
    print(f"  ROC AUC Score      : {auc_score:.4f}")
    print(f"  Accuracy           : {acc:.4f}")
    print(f"  1/FPR @ 50% TPR    : {bg_rejection_50:.2f}")
    print(f"  Artifacts saved to : {run_dir}/")
    print("==================================================\n")


def save_sf_circuit_diagram(prog, save_path, title="CV-QML Circuit"):
    """Renders a Strawberry Fields sf.Program into a clean PNG diagram."""
    n_modes = prog.num_subsystems

    # Color coding by CV gate family
    gate_colors = {
        "Dgate": "#D6EAF8",   # Displacement (Blue)
        "Sgate": "#E8DAEF",   # Squeezing (Purple)
        "Rgate": "#FCF3CF",   # Rotation (Yellow)
        "Kgate": "#FADBD8",   # Kerr (Red/Pink - non-Gaussian)
        "Vgate": "#FADBD8",   # Cubic phase (Red/Pink - non-Gaussian)
        "BSgate": "#D5F5E3",  # Beamsplitter (Green)
        "CXgate": "#D5F5E3",  # Controlled-X (Green)
        "CZgate": "#D5F5E3",  # Controlled-Z (Green)
    }

    # Track the next available column index for each mode
    mode_cursor = [0] * n_modes
    placed_ops = []  # (col, mode_indices, gate_name)

    for cmd in prog.circuit:
        gate_name = cmd.op.__class__.__name__
        modes = [reg.ind for reg in cmd.reg]

        # Span across all modes between min and max for multi-mode gates so wires don't overlap boxes
        span = range(min(modes), max(modes) + 1)
        col = max(mode_cursor[m] for m in span)

        placed_ops.append((col, modes, gate_name))
        for m in span:
            mode_cursor[m] = col + 1

    total_cols = max(mode_cursor) if mode_cursor else 1
    fig, ax = plt.subplots(figsize=(max(7, total_cols * 0.95), max(2.8, n_modes * 0.85)))

    # Draw horizontal qumode lines
    for m in range(n_modes):
        ax.plot([-0.4, total_cols - 0.3], [m, m], color="#333333", lw=1.5, zorder=1)
        ax.text(-0.55, m, f"q[{m}]", ha="right", va="center", fontsize=10, fontweight="bold", family="monospace")

    # Draw gates
    for col, modes, gate_name in placed_ops:
        short_label = gate_name.replace("gate", "")  # e.g., Dgate -> D, BSgate -> BS
        color = gate_colors.get(gate_name, "#EAECEE")

        if len(modes) == 1:
            m = modes[0]
            rect = patches.FancyBboxPatch(
                (col - 0.32, m - 0.28), 0.64, 0.56,
                boxstyle="round,pad=0.04", facecolor=color, edgecolor="#222222", lw=1.2, zorder=3
            )
            ax.add_patch(rect)
            ax.text(col, m, short_label, ha="center", va="center", fontsize=9.5, fontweight="bold", zorder=4)

        elif len(modes) == 2:
            m1, m2 = modes
            # Vertical coupler line connecting the two modes
            ax.plot([col, col], [m1, m2], color="#1E8449", lw=2.2, zorder=2)
            for m in (m1, m2):
                rect = patches.FancyBboxPatch(
                    (col - 0.30, m - 0.24), 0.60, 0.48,
                    boxstyle="round,pad=0.03", facecolor=color, edgecolor="#1E8449", lw=1.2, zorder=3
                )
                ax.add_patch(rect)
                ax.text(col, m, short_label, ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=4)

    ax.set_xlim(-1.2, total_cols - 0.1)
    ax.set_ylim(-0.6, n_modes - 0.4)
    ax.invert_yaxis()  # q[0] at the top
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    config = parse_terminal_args()

    # Reproducibility
    np.random.seed(config["seed"])
    tf.random.set_seed(config["seed"])

    print(f"--- Starting HPC Run with Config: {json.dumps(config)} ---")

    # 1. Load Data & Build Batched Pipeline
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_dataset(config)
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=len(x_train), seed=config["seed"])
        .batch(config["batch_size"], drop_remainder=False)
    )
    # config = parse_terminal_args()
    # draw_circuit(config, save_dir=config["save_dir"])
    # 2. Initialize Quantum Model & Optimizer
    params, readout_layer = initialize_model(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["lr"])
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # 3. Training Loop with Best-Checkpoint Tracking
    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_val_auc = -1.0
    best_snapshot = None

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()
        batch_losses = []

        for x_batch, y_batch in train_ds:
            loss = train_step(
                x_batch, y_batch, config, params, readout_layer, optimizer, loss_fn
            )
            batch_losses.append(float(loss.numpy()))

        # Quick validation check at the end of each epoch
        val_logits = predict_logits(x_val, config, params, readout_layer)
        val_loss = float(loss_fn(y_val, val_logits).numpy())
        val_probs = tf.sigmoid(val_logits).numpy().ravel()
        val_auc = float(roc_auc_score(y_val.ravel(), val_probs))
        train_loss = float(np.mean(batch_losses))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        # Save best weights in memory so final evaluation uses the peak checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_snapshot = (
                {k: v.numpy().copy() for k, v in params.items()},
                readout_layer.get_weights() if readout_layer is not None else None
            )

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{config['epochs']:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

    # 4. Restore Best Checkpoint & Run Final Metrics/ROC Generation
    if best_snapshot is not None:
        best_q_weights, best_dense_weights = best_snapshot
        for k, val in best_q_weights.items():
            params[k].assign(val)
        if readout_layer is not None and best_dense_weights is not None:
            readout_layer.set_weights(best_dense_weights)

    evaluate_and_save_results(x_test, y_test, history, config, params, readout_layer)