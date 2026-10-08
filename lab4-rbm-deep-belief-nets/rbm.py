from util import *


class RestrictedBoltzmannMachine:
    """
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    """

    def __init__(
        self,
        ndim_visible,
        ndim_hidden,
        is_bottom=False,
        image_size=[28, 28],
        is_top=False,
        n_labels=10,
        batch_size=10,
    ):
        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net.
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net (visible = features + labels).
          n_labels: Number of label categories.
          batch_size: Size of mini-batch.
        """

        self.ndim_visible = ndim_visible
        self.ndim_hidden = ndim_hidden
        self.is_bottom = is_bottom

        if is_bottom:
            self.image_size = image_size

        self.is_top = is_top

        if is_top:
            # FIX: was hard-coded to 10, ignoring the argument
            self.n_labels = n_labels

        self.batch_size = batch_size

        self.delta_bias_v = 0
        self.delta_weight_vh = 0
        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))
        self.weight_vh = np.random.normal(
            loc=0.0, scale=0.01, size=(self.ndim_visible, self.ndim_hidden)
        )
        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))

        self.delta_weight_v_to_h = 0
        self.delta_weight_h_to_v = 0

        self.weight_v_to_h = None
        self.weight_h_to_v = None

        self.learning_rate = 0.01
        self.momentum = 0.7

        # learning rate for the directed (wake-sleep) weights
        self.learning_rate_dir = 0.01

        self.print_period = 5000

        # NEW: reconstruction loss (mean squared error per pixel) for every iteration
        self.recon_losses = []

        self.rf = (
            {  # receptive-fields. Only applicable when visible layer is input data
                "period": 5000,  # iteration period to visualize
                "grid": [5, 5],  # size of the grid
                "ids": np.random.randint(
                    0, self.ndim_hidden, 25
                ),  # pick some random hidden units
            }
        )

        return

    def cd1(self, visible_trainset, n_iterations=10000):
        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print("learning CD1")

        n_samples = visible_trainset.shape[0]

        for it in range(n_iterations):

            batch = np.random.choice(n_samples, self.batch_size, replace=False)
            vis_batch = visible_trainset[batch]

            # Positive phase: v_0 (data/probabilities) -> h_0 (binary sample)
            h_0_prob, h_0 = self.get_h_given_v(vis_batch)

            # Negative phase: h_0 -> v_1 -> h_1
            v_1_prob, v_1 = self.get_v_given_h(h_0)
            # FIX: drive the hidden units with the reconstruction *probabilities*
            # (less sampling noise, as recommended in Hinton's practical guide).
            h_1_prob, h_1 = self.get_h_given_v(v_1_prob)

            # FIX: the final (k-th) step uses probabilities for both v and h (see lecture slides):
            #   v_0 : data, h_0 : binary sample, v_1 : probability, h_1 : probability
            self.update_params(vis_batch, h_0, v_1_prob, h_1_prob)

            # mean squared error per pixel/unit - comparable across layer sizes
            self.recon_losses.append(np.mean((vis_batch - v_1_prob) ** 2))

            # visualize once in a while when visible layer is input images
            if it % self.rf["period"] == 0 and self.is_bottom:
                viz_rf(
                    weights=self.weight_vh[:, self.rf["ids"]].reshape(
                        (self.image_size[0], self.image_size[1], -1)
                    ),
                    it=it,
                    grid=self.rf["grid"],
                )

            if it % self.print_period == 0:
                print(
                    "iteration=%7d recon_loss(MSE)=%.5f"
                    % (it, np.mean(self.recon_losses[-self.print_period :]))
                )

        return

    def update_params(self, v_0, h_0, v_k, h_k):
        """Update the weight and bias parameters (with momentum).

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """

        samples = v_0.shape[0]

        weight_gradient = (np.dot(v_0.T, h_0) - np.dot(v_k.T, h_k)) / samples
        bias_visible_gradient = np.mean(v_0 - v_k, axis=0)
        bias_hidden_gradient = np.mean(h_0 - h_k, axis=0)

        self.delta_bias_v = (
            self.momentum * self.delta_bias_v
            + self.learning_rate * bias_visible_gradient
        )
        self.delta_weight_vh = (
            self.momentum * self.delta_weight_vh + self.learning_rate * weight_gradient
        )
        self.delta_bias_h = (
            self.momentum * self.delta_bias_h
            + self.learning_rate * bias_hidden_gradient
        )

        self.bias_v += self.delta_bias_v
        self.weight_vh += self.delta_weight_vh
        self.bias_h += self.delta_bias_h

        return

    def get_h_given_v(self, visible_minibatch):
        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses undirected weight "weight_vh" and bias "bias_h"

        Returns:
           tuple ( p(h|v) , h)
        """
        assert self.weight_vh is not None

        h_given_v = sigmoid(np.dot(visible_minibatch, self.weight_vh) + self.bias_h)
        return h_given_v, sample_binary(h_given_v)

    def get_v_given_h(self, hidden_minibatch):
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"

        Returns:
           tuple ( p(v|h) , v)
        """

        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:
            # visible layer = [features | labels]
            support = np.dot(hidden_minibatch, self.weight_vh.T) + self.bias_v

            data_support = support[:, : -self.n_labels]
            label_support = support[:, -self.n_labels :]

            # features: sigmoid + Bernoulli sample
            v_data_prob = sigmoid(data_support)
            v_data_sample = sample_binary(v_data_prob)

            # labels: softmax, then a *proper* categorical sample
            # FIX: was argmax (deterministic), which is not sampling.
            v_label_prob = softmax(label_support)
            cum = np.cumsum(v_label_prob, axis=1)
            u = np.random.rand(n_samples, 1)
            idx = np.minimum((cum < u).sum(axis=1), self.n_labels - 1)
            v_label_sample = np.zeros_like(v_label_prob)
            v_label_sample[np.arange(n_samples), idx] = 1.0

            v_given_h = np.concatenate((v_data_prob, v_label_prob), axis=1)
            random_sampling = np.concatenate((v_data_sample, v_label_sample), axis=1)

        else:
            v_given_h = sigmoid(
                np.dot(hidden_minibatch, self.weight_vh.T) + self.bias_v
            )
            random_sampling = sample_binary(v_given_h)

        return v_given_h, random_sampling

    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    def untwine_weights(self):

        self.weight_v_to_h = np.copy(self.weight_vh)
        self.weight_h_to_v = np.copy(np.transpose(self.weight_vh))
        self.weight_vh = None

    def get_h_given_v_dir(self, visible_minibatch):
        """Compute p(h|v) and h ~ p(h|v) with directed weight "weight_v_to_h" and bias "bias_h" """

        assert self.weight_v_to_h is not None

        h_given_v = sigmoid(np.dot(visible_minibatch, self.weight_v_to_h) + self.bias_h)
        return h_given_v, sample_binary(h_given_v)

    def get_v_given_h_dir(self, hidden_minibatch):
        """Compute p(v|h) and v ~ p(v|h) with directed weight "weight_h_to_v" and bias "bias_v" """

        assert self.weight_h_to_v is not None

        if self.is_top:
            raise NotImplementedError(
                "This case should never be executed : when the RBM is a part of a DBN and is at the top, it will have not have directed connections."
            )

        v_given_h = sigmoid(np.dot(hidden_minibatch, self.weight_h_to_v) + self.bias_v)
        return v_given_h, sample_binary(v_given_h)

    def update_generate_params(self, inps, trgs, preds):
        """Update generative weight "weight_h_to_v" and bias "bias_v"

        Delta rule from the lab text: dw_ij ~ x_i (y_j - y~_j)

        Args:
           inps: activities of the upper layer (x_i), shape (batch, n_upper)
           trgs: activities of the lower layer from the wake phase (y_j), shape (batch, n_lower)
           preds: generative prediction (probabilities) of the lower layer (y~_j)
        """

        n = inps.shape[0]

        self.delta_weight_h_to_v = (
            self.learning_rate_dir * np.dot(inps.T, trgs - preds) / n
        )
        self.delta_bias_v = self.learning_rate_dir * np.mean(trgs - preds, axis=0)

        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v

        return

    def update_recognize_params(self, inps, trgs, preds):
        """Update recognition weight "weight_v_to_h" and bias "bias_h"

        Args:
           inps: activities of the lower layer from the sleep phase (x_j), shape (batch, n_lower)
           trgs: activities of the upper layer from the sleep phase (y_i), shape (batch, n_upper)
           preds: recognition prediction (probabilities) of the upper layer (y~_i)
        """

        n = inps.shape[0]

        self.delta_weight_v_to_h = (
            self.learning_rate_dir * np.dot(inps.T, trgs - preds) / n
        )
        self.delta_bias_h = self.learning_rate_dir * np.mean(trgs - preds, axis=0)

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h

        return
