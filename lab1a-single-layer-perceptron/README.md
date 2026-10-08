# Lab 1a – Single-Layer Perceptron

Binary classification of 2D data with a single-layer network, implemented from scratch in NumPy.

**Notebook:** [`lab1a_single_layer_perceptron.ipynb`](lab1a_single_layer_perceptron.ipynb)

## What's inside

**3.1.1 – Data generation.** Two Gaussian clouds of linearly separable 2D points.

**3.1.2 – Learning rules on separable data**
- Perceptron learning rule compared with the delta rule (Widrow-Hoff), both online, over several learning rates.
- Delta rule in sequential vs. batch mode, using the batch update `ΔW = -η (WX - T) Xᵀ`.
- Sensitivity to random weight initialisation over 10,000 trials.
- Removing the bias term, and why the decision boundary is then forced through the origin.

**3.1.3 – Non-linearly separable data**
- How each learning rule behaves when no linear boundary exists.
- A harder dataset where one class is bimodal, with four sub-sampling scenarios (random removal, removal from one class only, and skewed removal inside one class). For each one the notebook measures how class imbalance moves the decision boundary and changes per-class accuracy.

## Takeaways
- The perceptron rule stops once every point is classified correctly, so it converges on separable data but never settles on non-separable data. The delta rule minimises the MSE and always converges to a stable boundary.
- Batch mode gives smoother learning curves. Online mode can converge in fewer epochs but is noisier.
- Without a bias the boundary has to pass through the origin, so some separable datasets can no longer be classified.
- Unbalanced training sets bias the boundary towards the under-represented class.
