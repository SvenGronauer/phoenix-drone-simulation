from pickle import load
import unittest
import os
from phoenix_drone_simulation.utils.utils import SparseProductLayer, load_network_json
import numpy as np
import torch

class TestCSRLayer(unittest.TestCase):

    def load_models(self):
        if not hasattr(self, "test_model_path"):
            self.test_model_path = "test_csr_model.json"

        if not hasattr(self, "csr_net"):
            self.csr_net = load_network_json(self.test_model_path, force_dense_matrices=False)
        
        if not hasattr(self, "standard_net"):
            self.standard_net = load_network_json(self.test_model_path, force_dense_matrices=True)

    def check_test_model_is_available(self):
        self.assertTrue(os.path.isfile("test_csr_model.json"), msg="Cannot find the test model")

    def test_biases_are_loaded_correctly(self):
        self.load_models()
        for layer_idx, layer in enumerate(self.csr_net):
            if isinstance(layer, SparseProductLayer):
                self.assertTrue(torch.allclose(self.csr_net[layer_idx].bias, self.standard_net[layer_idx].bias), msg="The bias of the csr layer is different")

    def test_weights_are_loaded_correctly(self):
        self.load_models()
        for layer_idx, layer in enumerate(self.csr_net):
            if isinstance(layer, SparseProductLayer):
                csr_product_res = self.csr_net[layer_idx].sparse_matrices[0].to_dense()
                for csr_mat in self.csr_net[layer_idx].sparse_matrices[1:]:
                    csr_product_res = csr_product_res @ csr_mat.to_dense()
                self.assertTrue(torch.allclose(csr_product_res, self.standard_net[layer_idx].weight), msg="The weights of the csr layer are different")

    def test_for_random_inputs(self):
        self.load_models()
        random_inputs = torch.Tensor(np.random.uniform(-1, 1, size=(100, 40)))

        csr_net_output = self.csr_net(random_inputs)
        standard_net_output = self.standard_net(random_inputs)

        self.assertTrue(torch.allclose(csr_net_output, standard_net_output, atol=1e-6), msg="Net produces the wrong results for random inputs")
        self.assertTrue(len(csr_net_output.shape) == len(standard_net_output.shape), msg="The output has the wrong shape (for multiple random inputs)")

    def test_for_single_random_input(self):
        self.load_models()
        random_inputs = torch.Tensor(np.random.uniform(-1, 1, size=(40,)))

        csr_net_output = self.csr_net(random_inputs)
        standard_net_output = self.standard_net(random_inputs)

        self.assertTrue(torch.allclose(csr_net_output, standard_net_output, atol=1e-6), msg="Net produces the wrong results for random inputs")
        self.assertTrue(len(csr_net_output.shape) == len(standard_net_output.shape), msg="The output has the wrong shape (for a single random input)")

if __name__ == '__main__':
    unittest.main()
