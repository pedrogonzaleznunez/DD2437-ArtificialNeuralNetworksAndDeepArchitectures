# Lab 2 – RBF Networks, Competitive Learning and Self-Organising Maps

**Notebook:** [`lab2_rbf_cl_som.ipynb`](lab2_rbf_cl_som.ipynb) · **Data:** [`data_lab2/`](data_lab2/)

## Part I – Radial Basis Function networks

**3.1 – Least-squares training.** Approximating `sin(2x)` and the square wave `sgn(sin(2x))` with Gaussian RBF units, and finding how many units each residual-error threshold needs.

**3.2 – Regression with noise.** Online delta-rule training on noisy data, with sweeps over unit count, kernel width and learning rate. Compares uniform vs. random centre placement and benchmarks the RBF network against an MLP.

**3.3 – Competitive learning (CL) for placing RBF centres**
- CL-placed centres compared with manual and random placement, on clean and noisy data.
- The *dead units* problem, and how a leaky CL variant fixes it (6 dead units drop to 0, and test error falls from 0.139 to 0.039).
- A 2D→2D RBF network on the ballistic dataset, mapping (angle, velocity) to (distance, height).

## Part II – Self-Organising Maps

**4.1 – Topological ordering of animal species.** A 1D SOM with 100 nodes orders 32 animals by 84 binary attributes. Similar species (insects, birds, big mammals…) end up next to each other.

**4.2 – Cyclic tour.** A circular 1D SOM finds a short closed route through 10 cities, a travelling-salesman-style heuristic.

**4.3 – Votes of Swedish MPs.** A 10×10 SOM over the voting records of 349 members of parliament on 31 votes, coloured by party, sex and district to see which groupings show up.
