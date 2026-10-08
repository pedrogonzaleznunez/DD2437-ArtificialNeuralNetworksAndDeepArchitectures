# Lab 3 – Hopfield Networks

Auto-associative memory with Hopfield networks, implemented from scratch in NumPy.

**Notebook:** [`lab3_hopfield_networks.ipynb`](lab3_hopfield_networks.ipynb) · **Data:** [`data_lab3/pict.dat`](data_lab3/pict.dat) (eleven 32×32 binary images)

## What's inside

- **3.1 – Convergence and attractors.** Hebbian learning of three 8-bit patterns, then recall from distorted inputs. An exhaustive search over all 256 states finds 14 attractors. Heavily distorted inputs end up in spurious states or in inverted patterns (`-x`).
- **3.2 – Sequential update.** Three 1024-pixel images stored and recalled from a degraded image (p10) and a mixture of two images (p11). Asynchronous random-unit updates are shown converging step by step.
- **3.3 – Energy.** The energy function evaluated at stored patterns and at distorted ones, and how it falls during sequential recall. The same is done with a random Gaussian weight matrix for comparison.
- **3.4 – Distortion resistance.** Recall accuracy as a growing percentage of pixels is flipped (0–100%). The code is in the notebook but commented out because it takes very long to run.
- **3.5 – Capacity.** Stability of the stored images as more are added, then capacity with random patterns: with and without noise, and with biased (non-zero-mean) patterns.
- **3.6 – Sparse patterns.** Binary {0, 1} sparse patterns stored with a threshold (bias) term, and capacity as a function of the bias.

## Results
- All three stored images are stable. With a 4th image stored, **none** of them are stable any more, because the pictures are strongly correlated.
- With 1024 neurons and random uncorrelated patterns, 50 out of 50 stored patterns stay stable, and about 75 out of 80 are still stable. That is far beyond the capacity reached with the real images.
- Biased (non-zero-mean) random patterns lower the capacity a lot. Sparse patterns with a suitable threshold bring it back up.
