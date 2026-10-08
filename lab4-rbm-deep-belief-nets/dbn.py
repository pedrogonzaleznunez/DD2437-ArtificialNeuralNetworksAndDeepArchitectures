import os

from util import *
from rbm import RestrictedBoltzmannMachine


class DeepBeliefNet:
    """
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis]
                               `-> [lbl]
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    """

    def __init__(self, sizes, image_size, n_labels, batch_size):
        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {
            "vis--hid": RestrictedBoltzmannMachine(
                ndim_visible=sizes["vis"],
                ndim_hidden=sizes["hid"],
                is_bottom=True,
                image_size=image_size,
                batch_size=batch_size,
            ),
            "hid--pen": RestrictedBoltzmannMachine(
                ndim_visible=sizes["hid"],
                ndim_hidden=sizes["pen"],
                batch_size=batch_size,
            ),
            "pen+lbl--top": RestrictedBoltzmannMachine(
                ndim_visible=sizes["pen"] + sizes["lbl"],
                ndim_hidden=sizes["top"],
                is_top=True,
                n_labels=n_labels,
                batch_size=batch_size,
            ),
        }

        self.sizes = sizes
        self.image_size = image_size
        self.batch_size = batch_size

        self.n_gibbs_recog = 15
        self.n_gibbs_gener = 200
        self.n_gibbs_wakesleep = 5

        self.print_period = 2000

        return

    def recognize(self, true_img, true_lbl, chunk=1000):
        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_img: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
          chunk: number of samples processed at once (keeps memory use low)

        Returns:
          accuracy in percent
        """

        n_samples = true_img.shape[0]
        n_lbl = self.sizes["lbl"]
        predicted_lbl = np.zeros(true_lbl.shape)

        # FIX: process in chunks - the 2000-unit top layer for 60k images at once needs several GB.
        for start in range(0, n_samples, chunk):
            vis = true_img[start : start + chunk]

            # start the net by telling it that it knows nothing about the labels
            lbl = np.ones((vis.shape[0], n_lbl)) / float(n_lbl)

            # 1. Drive the network bottom -> top with the directed recognition weights.
            #    Probabilities are passed up (same representation the next RBM was trained on).
            hid_prob, _ = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis)
            pen_prob, _ = self.rbm_stack["hid--pen"].get_h_given_v_dir(hid_prob)

            # 2. Alternating Gibbs sampling in the top RBM; image features stay fixed,
            #    label units are free and read out through the softmax.
            for _ in range(self.n_gibbs_recog):
                top_vis = np.concatenate((pen_prob, lbl), axis=1)
                _, top_hid_sample = self.rbm_stack["pen+lbl--top"].get_h_given_v(
                    top_vis
                )
                top_vis_prob, _ = self.rbm_stack["pen+lbl--top"].get_v_given_h(
                    top_hid_sample
                )
                lbl = top_vis_prob[:, -n_lbl:]

            predicted_lbl[start : start + chunk] = lbl

        accuracy = 100.0 * np.mean(
            np.argmax(predicted_lbl, axis=1) == np.argmax(true_lbl, axis=1)
        )
        print("accuracy = %.2f%%" % accuracy)

        return accuracy

    def generate(self, true_lbl, name):
        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """
        n_sample = true_lbl.shape[0]
        n_lbl = self.sizes["lbl"]
        top = self.rbm_stack["pen+lbl--top"]
        hp = self.rbm_stack["hid--pen"]
        vh = self.rbm_stack["vis--hid"]

        records = []
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])

        lbl = true_lbl  # clamped in ALL iterations

        # init the 500 units with a *binary sample from the biases* (slide option 2)
        p0 = sigmoid(top.bias_v[:-n_lbl])
        pen = sample_binary(np.tile(p0, (n_sample, 1)))

        for _ in range(self.n_gibbs_gener):
            # top RBM Gibbs step: [500 binary | label clamped] <-> 2000 binary
            _, top_hid = top.get_h_given_v(np.concatenate((pen, lbl), axis=1))
            _, top_vis_sample = top.get_v_given_h(top_hid)
            pen = top_vis_sample[:, :-n_lbl]            # 500 units: BINARY sample

            # propagate down with generative weights (binary sampling), image = probabilities
            _, hid = hp.get_v_given_h_dir(pen)          # binary sample
            vis_prob, _ = vh.get_v_given_h_dir(hid)     # probabilities -> image
            records.append([ax.imshow(vis_prob.reshape(self.image_size), cmap="gray",
                                      vmin=0, vmax=1, animated=True, interpolation=None)])

        stitch_video(fig, records).save("%s.generate%d.mp4" % (name, np.argmax(true_lbl)))
        plt.close(fig)

    @staticmethod
    def _all_exist(paths):
        return all(os.path.exists(p) for p in paths)

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):
        """
        Greedy layer-wise training by stacking RBMs. Loads saved parameters if ALL of them exist,
        otherwise learns layer-by-layer.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        loc = "trained_rbm"
        names = ["vis--hid", "hid--pen", "pen+lbl--top"]
        files = [
            "%s/rbm.%s.%s.npy" % (loc, n, p)
            for n in names
            for p in ("weight_vh", "bias_v", "bias_h")
        ]

        # FIX: check first, then load everything or nothing. Previously a partial load untwined
        # weights, and the retraining then crashed on weight_vh being None.
        if self._all_exist(files):
            self.loadfromfile_rbm(loc=loc, name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()

            self.loadfromfile_rbm(loc=loc, name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()

            self.loadfromfile_rbm(loc=loc, name="pen+lbl--top")
            return

        os.makedirs(loc, exist_ok=True)

        print("training vis--hid")
        self.rbm_stack["vis--hid"].cd1(vis_trainset, n_iterations)
        self.savetofile_rbm(loc=loc, name="vis--hid")
        self.rbm_stack["vis--hid"].untwine_weights()

        # pass probabilities up as training data for the next RBM
        hid_trainset, _ = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_trainset)

        print("training hid--pen")
        self.rbm_stack["hid--pen"].cd1(hid_trainset, n_iterations)
        self.savetofile_rbm(loc=loc, name="hid--pen")
        self.rbm_stack["hid--pen"].untwine_weights()

        pen_trainset, _ = self.rbm_stack["hid--pen"].get_h_given_v_dir(hid_trainset)

        print("training pen+lbl--top")
        top_trainset = np.concatenate((pen_trainset, lbl_trainset), axis=1)

        # the top RBM keeps its undirected weights
        self.rbm_stack["pen+lbl--top"].cd1(top_trainset, n_iterations)
        self.savetofile_rbm(loc=loc, name="pen+lbl--top")

        return

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):
        """
        Wake-sleep method for learning all the parameters of network.
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print("\ntraining wake-sleep..")

        loc = "trained_dbn"
        files = [
            "%s/dbn.%s.%s.npy" % (loc, n, p)
            for n in ("vis--hid", "hid--pen")
            for p in ("weight_v_to_h", "weight_h_to_v", "bias_v", "bias_h")
        ] + [
            "%s/rbm.pen+lbl--top.%s.npy" % (loc, p)
            for p in ("weight_vh", "bias_v", "bias_h")
        ]

        if self._all_exist(files):
            self.loadfromfile_dbn(loc=loc, name="vis--hid")
            self.loadfromfile_dbn(loc=loc, name="hid--pen")
            self.loadfromfile_rbm(loc=loc, name="pen+lbl--top")
            return

        os.makedirs(loc, exist_ok=True)

        vh = self.rbm_stack["vis--hid"]
        hp = self.rbm_stack["hid--pen"]
        top = self.rbm_stack["pen+lbl--top"]
        n_lbl = self.sizes["lbl"]

        self.n_samples = vis_trainset.shape[0]

        for it in range(n_iterations):

            batch = np.random.choice(self.n_samples, self.batch_size, replace=False)
            vis_batch = vis_trainset[batch]
            lbl_batch = lbl_trainset[batch]

            # ---------------- wake phase: bottom -> top with recognition weights ----------------
            _, wake_hid = vh.get_h_given_v_dir(vis_batch)
            _, wake_pen = hp.get_h_given_v_dir(wake_hid)

            # ---------------- Gibbs sampling in the top RBM (labels visible, not clamped after step 0) ----
            v_0 = np.concatenate((wake_pen, lbl_batch), axis=1)
            _, h_0 = top.get_h_given_v(v_0)  # statistics for the positive phase

            h = h_0
            for _ in range(self.n_gibbs_wakesleep):
                v_prob, v_sample = top.get_v_given_h(h)
                h_prob, h = top.get_h_given_v(v_sample)

            # ---------------- sleep phase: top -> bottom with generative weights ----------------
            sleep_pen = v_sample[:, :-n_lbl]
            _, sleep_hid = hp.get_v_given_h_dir(sleep_pen)
            _, sleep_vis = vh.get_v_given_h_dir(sleep_hid)

            # ---------------- predictions (do not change the activities) ----------------
            # generative predictions from wake activities
            pred_wake_hid, _ = hp.get_v_given_h_dir(wake_pen)
            pred_wake_vis, _ = vh.get_v_given_h_dir(wake_hid)
            # recognition predictions from sleep activities
            pred_sleep_pen, _ = hp.get_h_given_v_dir(sleep_hid)
            pred_sleep_hid, _ = vh.get_h_given_v_dir(sleep_vis)

            # ---------------- parameter updates ----------------
            # generative weights (learned from wake activities)
            hp.update_generate_params(wake_pen, wake_hid, pred_wake_hid)
            vh.update_generate_params(wake_hid, vis_batch, pred_wake_vis)

            # top RBM: CD with the wake-phase start and the k-step reconstruction (probabilities)
            top.update_params(v_0, h_0, v_prob, h_prob)

            # recognition weights (learned from sleep activities)
            hp.update_recognize_params(sleep_hid, sleep_pen, pred_sleep_pen)
            vh.update_recognize_params(sleep_vis, sleep_hid, pred_sleep_hid)

            if it % self.print_period == 0:
                print("iteration=%7d" % it)

        self.savetofile_dbn(loc=loc, name="vis--hid")
        self.savetofile_dbn(loc=loc, name="hid--pen")
        self.savetofile_rbm(loc=loc, name="pen+lbl--top")

        return

    def loadfromfile_rbm(self, loc, name):

        self.rbm_stack[name].weight_vh = np.load(
            "%s/rbm.%s.weight_vh.npy" % (loc, name)
        )
        self.rbm_stack[name].bias_v = np.load("%s/rbm.%s.bias_v.npy" % (loc, name))
        self.rbm_stack[name].bias_h = np.load("%s/rbm.%s.bias_h.npy" % (loc, name))
        print("loaded rbm[%s] from %s" % (name, loc))
        return

    def savetofile_rbm(self, loc, name):

        os.makedirs(loc, exist_ok=True)
        np.save("%s/rbm.%s.weight_vh" % (loc, name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v" % (loc, name), self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h" % (loc, name), self.rbm_stack[name].bias_h)
        return

    def loadfromfile_dbn(self, loc, name):

        self.rbm_stack[name].weight_v_to_h = np.load(
            "%s/dbn.%s.weight_v_to_h.npy" % (loc, name)
        )
        self.rbm_stack[name].weight_h_to_v = np.load(
            "%s/dbn.%s.weight_h_to_v.npy" % (loc, name)
        )
        self.rbm_stack[name].bias_v = np.load("%s/dbn.%s.bias_v.npy" % (loc, name))
        self.rbm_stack[name].bias_h = np.load("%s/dbn.%s.bias_h.npy" % (loc, name))
        print("loaded rbm[%s] from %s" % (name, loc))
        return

    def savetofile_dbn(self, loc, name):

        os.makedirs(loc, exist_ok=True)
        np.save(
            "%s/dbn.%s.weight_v_to_h" % (loc, name), self.rbm_stack[name].weight_v_to_h
        )
        np.save(
            "%s/dbn.%s.weight_h_to_v" % (loc, name), self.rbm_stack[name].weight_h_to_v
        )
        np.save("%s/dbn.%s.bias_v" % (loc, name), self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h" % (loc, name), self.rbm_stack[name].bias_h)
        return
