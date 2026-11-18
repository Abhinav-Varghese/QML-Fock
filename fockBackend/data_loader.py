import gc
import h5py
import numpy as np
import os


class TTBarZJetsData:
    """
    Supervised data loader for TTBar vs. ZJetsToNuNu HDF5 jet datasets.
    - Keeps top N_CONSTITUENTS particles per jet sorted by descending pT.
    - Drops jets with fewer than N_CONSTITUENTS valid particles.
    - Splits into deterministic val/test sets and seeded training sets (50/50 sig/bkg).
    - Clips |eta| and |phi| to [-1, 1] and scales features for Qumode encoding.
    """

    N_CONSTITUENTS = 8

    def __init__(self, cfg):
        self.cfg = cfg
        dataset_path = os.path.join("/rds/general/user/av523/home/projects/myproject/data/QMLoutputs/TTBar_+ZJetsToNuNU_flat_merged.h5")
        self.const, self.truth, self.jet_pt = self.load_data(dataset_path)
        self._filter_low_multiplicity_jets()

        self.n = cfg.circuit.wires
        if self.n > self.N_CONSTITUENTS:
            raise ValueError(
                f"Requested {self.n} wires/constituents, but data is cut to {self.N_CONSTITUENTS}."
            )

        self.n_train = cfg.data.train_num_samples
        self.n_val = cfg.data.val_num_samples
        self.n_test = cfg.data.test_num_samples

    def load_data(self, path):
        with h5py.File(path, "r") as f:
            const = f["jetConstituentsList"][...].astype(np.float64)
            truth = np.rint(f["truth_labels"][...]).astype(np.int64)
            names = [
                x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
                for x in f["particleFeatureNames"][()]
            ]

        self.eta_idx = names.index("eta")
        self.phi_idx = names.index("phi")
        self.pt_idx = names.index("pt")

        print("\nInitial Feature Limits")
        for name, idx in [("eta", self.eta_idx), ("phi", self.phi_idx), ("pt", self.pt_idx)]:
            values = const[:, :, idx]
            finite = values[np.isfinite(values)]
            if finite.size:
                print(
                    f"{name}: min={finite.min():.6f}, max={finite.max():.6f}, "
                    f"p1={np.percentile(finite, 1):.6f}, p99={np.percentile(finite, 99):.6f}"
                )

        # Sort each jet by descending pt and keep top N_CONSTITUENTS
        pt_all = const[:, :, self.pt_idx]
        key = np.where(np.isfinite(pt_all) & (pt_all > 0), pt_all, -np.inf)
        order = np.argsort(-key, axis=1)[:, : self.N_CONSTITUENTS]
        const = np.take_along_axis(const, order[:, :, None], axis=1)

        # Total jet pt across the kept constituents
        constituent_pt = const[:, :, self.pt_idx]
        valid = np.isfinite(constituent_pt) & (constituent_pt > 0)
        jet_pt = np.where(valid, constituent_pt, 0.0).sum(axis=1)

        return const, truth, jet_pt

    def _filter_low_multiplicity_jets(self):
        pt = self.const[:, :, self.pt_idx]
        eta = self.const[:, :, self.eta_idx]
        phi = self.const[:, :, self.phi_idx]
        valid = np.isfinite(pt) & (pt > 0) & np.isfinite(eta) & np.isfinite(phi)
        keep = valid.sum(axis=1) >= self.N_CONSTITUENTS

        n_dropped = int((~keep).sum())
        n_total = len(keep)
        print(
            f"[TTBarZJetsData] Dropping {n_dropped}/{n_total} jets "
            f"({n_dropped / n_total * 100:.3f}%) with < {self.N_CONSTITUENTS} valid constituents."
        )

        self.const = self.const[keep]
        self.truth = self.truth[keep]
        self.jet_pt = self.jet_pt[keep]

    def build_X(self, const, jet_pt, indices, k):
        c = const[indices, :k]
        eta = c[:, :, self.eta_idx]
        pt_frac = c[:, :, self.pt_idx] / (jet_pt[indices, None] + 1e-12)
        phi = c[:, :, self.phi_idx]
        return np.stack([eta, pt_frac, phi], axis=-1)

    @staticmethod
    def global_shuffle(indices, seed):
        rng = np.random.default_rng(seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        return shuffled

    def split_train_val_test(self, rng):
        bkg = np.where(self.truth == self.truth.min())[0]
        sig = np.where(self.truth == self.truth.max())[0]
        global_seed = 20

        bkg = self.global_shuffle(bkg, global_seed)
        sig = self.global_shuffle(sig, global_seed)

        n_val_each = self.n_val // 2
        n_test_each = self.n_test // 2
        n_train_each = self.n_train // 2

        te_bkg = bkg[-n_test_each:]
        te_sig = sig[-n_test_each:]
        vl_bkg = bkg[-(n_test_each + n_val_each) : -n_test_each]
        vl_sig = sig[-(n_test_each + n_val_each) : -n_test_each]

        remaining_bkg = bkg[: -(n_test_each + n_val_each)]
        remaining_sig = sig[: -(n_test_each + n_val_each)]

        if n_train_each > len(remaining_bkg) or n_train_each > len(remaining_sig):
            raise ValueError(
                f"Requested train_num_samples={self.n_train} ({n_train_each} per class) "
                f"but only {len(remaining_bkg)} bkg / {len(remaining_sig)} sig remain."
            )

        tr_bkg = rng.permutation(remaining_bkg)[:n_train_each]
        tr_sig = rng.permutation(remaining_sig)[:n_train_each]

        tr_idx = self.global_shuffle(np.concatenate([tr_bkg, tr_sig]), global_seed)
        vl_idx = self.global_shuffle(np.concatenate([vl_bkg, vl_sig]), global_seed)
        te_idx = self.global_shuffle(np.concatenate([te_bkg, te_sig]), global_seed)

        return tr_idx, vl_idx, te_idx

    def normalisation(self, X):
        eta = np.clip(X[:, :, 0], -1.0, 1.0)
        pt_frac = X[:, :, 1]
        phi = np.clip(X[:, :, 2], -1.0, 1.0)

        out = np.empty(X.shape[:2] + (3,), dtype=np.float32)
        out[:, :, 0] = self.cfg.data.s_mult * eta
        out[:, :, 1] = self.cfg.data.d_mult * pt_frac
        out[:, :, 2] = phi * np.pi
        return out

    def free_raw_data(self):
        for attr in ("const", "truth", "jet_pt"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()

    def createSupervisedDatasets(self, rng):
        tr_idx, vl_idx, te_idx = self.split_train_val_test(rng)

        all_idx = np.unique(np.concatenate([tr_idx, vl_idx, te_idx]))
        X_all = self.build_X(self.const, self.jet_pt, all_idx, k=self.n)
        r2i = {int(i): r for r, i in enumerate(all_idx)}

        def gather(idx):
            return X_all[np.array([r2i[int(i)] for i in idx])]

        X_tr = self.normalisation(gather(tr_idx))
        X_vl = self.normalisation(gather(vl_idx))
        X_te = self.normalisation(gather(te_idx))

        sig_label = self.truth.max()
        y_tr = (self.truth[tr_idx] == sig_label).astype(np.float32)[:, None]
        y_vl = (self.truth[vl_idx] == sig_label).astype(np.float32)[:, None]
        y_te = (self.truth[te_idx] == sig_label).astype(np.float32)[:, None]

        del X_all, r2i
        gc.collect()

        return X_tr, y_tr, X_vl, y_vl, X_te, y_te