"""
Lab 4 - RBM experiments
  4.1 C : receptive fields + fidelity of the reconstructions of the 784-500 RBM
  4.2 A : greedy stack 784-500-500, reconstruction loss of both RBMs

Uses rbm.py as is (cd1, get_h_given_v, ...). Run from the lab folder:
    python experiments_rf_and_stack.py
Figures and numbers go to results/.
"""
import os
import time
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from util import read_mnist
from rbm import RestrictedBoltzmannMachine

np.random.seed(0)
OUT = "results"
os.makedirs(OUT, exist_ok=True)

IMG = [28, 28]
BATCH = 20
EPOCHS = 15
N_TRAIN = 60000
IT_PER_EPOCH = N_TRAIN // BATCH  # 3000 iterations = one sweep through the training set
RF_SNAPSHOTS = [0, 1, 5, EPOCHS]  # epochs at which receptive fields are stored

train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=IMG, n_train=N_TRAIN, n_test=10000)


# ----------------------------------------------------------------------------- helpers
def make_rbm(n_vis, n_hid, bottom):
    rbm = RestrictedBoltzmannMachine(ndim_visible=n_vis, ndim_hidden=n_hid, is_bottom=bottom,
                                     image_size=IMG, batch_size=BATCH)
    rbm.rf["period"] = 10 ** 9  # we draw our own RF figures
    rbm.print_period = 10 ** 9  # we print once per epoch instead
    return rbm


def recon_mse(rbm, data):
    """Mean squared error per unit of one deterministic up-down pass v -> p(h|v) -> p(v|h)."""
    h_prob, _ = rbm.get_h_given_v(data)
    v_prob, _ = rbm.get_v_given_h(h_prob)
    return float(np.mean((data - v_prob) ** 2)), v_prob


def train(rbm, train_set, test_set, name, rf_store=None):
    """Train epoch by epoch with cd1 and record train/test reconstruction loss."""
    hist = {"epoch": [0], "test_mse": [recon_mse(rbm, test_set)[0]], "w_absmax": [float(np.abs(rbm.weight_vh).max())]}
    if rf_store is not None:
        rf_store[0] = rbm.weight_vh.copy()
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        rbm.cd1(train_set, n_iterations=IT_PER_EPOCH)
        mse = recon_mse(rbm, test_set)[0]
        hist["epoch"].append(ep)
        hist["test_mse"].append(mse)
        hist["w_absmax"].append(float(np.abs(rbm.weight_vh).max()))
        if rf_store is not None and ep in RF_SNAPSHOTS:
            rf_store[ep] = rbm.weight_vh.copy()
        print("[%s] epoch %2d  train_mse(last epoch)=%.5f  test_mse=%.5f  (%.0fs)"
              % (name, ep, np.mean(rbm.recon_losses[-IT_PER_EPOCH:]), mse, time.time() - t0))
    hist["train_mse_per_it"] = rbm.recon_losses
    hist["time_s"] = time.time() - t0
    return hist


def show_rf(ax, w, title=None):
    m = np.abs(w).max()  # symmetric colour scale per unit: red = positive, blue = negative
    ax.imshow(w.reshape(IMG), cmap="bwr", vmin=-m, vmax=m)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


def smooth(x, k=500):
    return np.convolve(x, np.ones(k) / k, mode="valid")


# ============================================================================ 4.1 C
print("\n===== RBM 1 : 784 - 500 =====")
rbm1 = make_rbm(784, 500, bottom=True)
rf_hist = {}
hist1 = train(rbm1, train_imgs, test_imgs, "vis--hid", rf_store=rf_hist)
W = rbm1.weight_vh

# (a) final receptive fields of 25 random hidden units
units = np.random.choice(500, 25, replace=False)
fig, axs = plt.subplots(5, 5, figsize=(6, 6.4))
for ax, u in zip(axs.flat, units):
    show_rf(ax, W[:, u], "unit %d" % u)
fig.suptitle("Receptive fields after %d epochs (random units)" % EPOCHS)
plt.tight_layout(); plt.savefig("%s/41C_rf_random_units.png" % OUT, dpi=150); plt.close()

# (b) the same 8 units across training: how the fields develop
track = units[:8]
fig, axs = plt.subplots(len(RF_SNAPSHOTS), len(track), figsize=(1.2 * len(track), 1.35 * len(RF_SNAPSHOTS)))
for r, ep in enumerate(RF_SNAPSHOTS):
    for c, u in enumerate(track):
        show_rf(axs[r, c], rf_hist[ep][:, u], "ep %d" % ep if c == 0 else None)
fig.suptitle("Development of receptive fields (rows = epochs)")
plt.tight_layout(); plt.savefig("%s/41C_rf_evolution.png" % OUT, dpi=150); plt.close()

# (c) which digits drive each unit most: mean image of the 100 test digits with highest p(h|v)
h_test, _ = rbm1.get_h_given_v(test_imgs)
fig, axs = plt.subplots(2, 8, figsize=(9.6, 2.8))
for c, u in enumerate(track):
    top = np.argsort(-h_test[:, u])[:100]
    show_rf(axs[0, c], W[:, u], "unit %d" % u)
    axs[1, c].imshow(test_imgs[top].mean(0).reshape(IMG), cmap="gray")
    axs[1, c].set_xticks([]); axs[1, c].set_yticks([])
    digits = np.argmax(test_lbls[top], 1)
    axs[1, c].set_title("most: %d (%d%%)" % (np.bincount(digits, minlength=10).argmax(),
                                             np.bincount(digits, minlength=10).max()), fontsize=7)
axs[0, 0].set_ylabel("weights"); axs[1, 0].set_ylabel("top-100\ninputs")
plt.tight_layout(); plt.savefig("%s/41C_rf_vs_preferred_inputs.png" % OUT, dpi=150); plt.close()

# (d) weight distribution (Q&A Q7: roughly within -5..5)
plt.figure(figsize=(5, 3))
plt.hist(W.ravel(), bins=200)
plt.xlabel("weight value"); plt.ylabel("count"); plt.title("RBM 1 weight distribution")
plt.tight_layout(); plt.savefig("%s/41C_weight_hist.png" % OUT, dpi=150); plt.close()

# (e) fidelity of reconstructions: original / p(v|h) / one binary Gibbs step
idx = np.array([np.where(np.argmax(test_lbls, 1) == d)[0][0] for d in range(10)])
v = test_imgs[idx]
h_p, h_s = rbm1.get_h_given_v(v)
v_det, _ = rbm1.get_v_given_h(h_p)   # deterministic reconstruction
v_sto, _ = rbm1.get_v_given_h(h_s)   # reconstruction from a binary hidden sample (what CD-1 sees)
fig, axs = plt.subplots(3, 10, figsize=(10, 3.3))
for c in range(10):
    for r, img in enumerate([v[c], v_det[c], v_sto[c]]):
        axs[r, c].imshow(img.reshape(IMG), cmap="gray", vmin=0, vmax=1)
        axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
for r, lab in enumerate(["original", "recon p(h)", "recon h~"]):
    axs[r, 0].set_ylabel(lab, fontsize=8)
plt.tight_layout(); plt.savefig("%s/41C_reconstructions.png" % OUT, dpi=150); plt.close()

# per-digit reconstruction error on the whole test set
_, v_rec = recon_mse(rbm1, test_imgs)
err = np.mean((test_imgs - v_rec) ** 2, axis=1)
per_digit = {d: float(err[np.argmax(test_lbls, 1) == d].mean()) for d in range(10)}

# ============================================================================ 4.2 A
print("\n===== RBM 2 : 500 - 500 (trained on p(h|v) of RBM 1) =====")
rbm1.untwine_weights()  # as in dbn.train_greedylayerwise: RBM 1 is frozen, connections become directed
hid_train, _ = rbm1.get_h_given_v_dir(train_imgs)
hid_test, _ = rbm1.get_h_given_v_dir(test_imgs)

rbm2 = make_rbm(500, 500, bottom=False)
hist2 = train(rbm2, hid_train, hid_test, "hid--pen")

# reconstruction loss curves of both RBMs
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
for h, lab in [(hist1, "RBM 1 (784-500), pixels"), (hist2, "RBM 2 (500-500), hidden units of RBM 1")]:
    tr = smooth(np.array(h["train_mse_per_it"]))
    axs[0].plot(np.arange(len(tr)) / IT_PER_EPOCH, tr, label=lab)
    axs[1].plot(h["epoch"], h["test_mse"], "o-", label=lab)
axs[0].set_title("training MSE per unit (moving avg, CD-1 minibatches)")
axs[1].set_title("test MSE per unit, deterministic up-down pass")
for ax in axs:
    ax.set_xlabel("epoch"); ax.set_yscale("log"); ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("%s/42A_recon_loss_both_rbms.png" % OUT, dpi=150); plt.close()

# pixel-space reconstruction through the whole stack: v -> h1 -> h2 -> h1' -> v'
h2_p, _ = rbm2.get_h_given_v(hid_test)
h1_rec, _ = rbm2.get_v_given_h(h2_p)
v_stack, _ = rbm1.get_v_given_h_dir(h1_rec)
stack_mse = float(np.mean((test_imgs - v_stack) ** 2))

fig, axs = plt.subplots(3, 10, figsize=(10, 3.3))
for c, i in enumerate(idx):
    for r, img in enumerate([test_imgs[i], v_rec[i], v_stack[i]]):
        axs[r, c].imshow(img.reshape(IMG), cmap="gray", vmin=0, vmax=1)
        axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
for r, lab in enumerate(["original", "via RBM 1", "via RBM 1+2"]):
    axs[r, 0].set_ylabel(lab, fontsize=8)
plt.tight_layout(); plt.savefig("%s/42A_stack_reconstructions.png" % OUT, dpi=150); plt.close()

# ----------------------------------------------------------------------------- summary
summary = {
    "settings": {"batch": BATCH, "epochs": EPOCHS, "lr": rbm1.learning_rate, "momentum": rbm1.momentum},
    "rbm1_test_mse_final": hist1["test_mse"][-1],
    "rbm2_test_mse_final": hist2["test_mse"][-1],
    "stack_pixel_mse_v_to_h2_to_v": stack_mse,
    "rbm1_test_mse_per_epoch": hist1["test_mse"],
    "rbm2_test_mse_per_epoch": hist2["test_mse"],
    "rbm1_max_abs_weight": hist1["w_absmax"][-1],
    "rbm2_max_abs_weight": hist2["w_absmax"][-1],
    "rbm1_mean_hidden_activation": float(h_test.mean()),
    "rbm1_per_digit_mse": per_digit,
    "train_time_s": {"rbm1": hist1["time_s"], "rbm2": hist2["time_s"]},
}
with open("%s/summary.json" % OUT, "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps({k: v for k, v in summary.items() if "per_epoch" not in k}, indent=2))