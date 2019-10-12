"""Unit tests for dtreeviz.shadow.ShadowDecTree"""

import unittest

from joblib import load

from dtreeviz.shadow import ShadowDecTree


class TestShadowDecTree(unittest.TestCase):

    def setUp(self) -> None:
        self.tree_model = load("./fixtures/decision_tree_classifier.joblib")

    def test_get_node_type(self):
        node_types = ShadowDecTree.get_node_type(self.tree_model)

        total_nodes = 49
        total_leaf_nodes = 25
        total_split_nodes = total_nodes - total_leaf_nodes
        leaf_node_ids = [4, 5, 8, 9, 11, 12, 16, 17, 19, 20, 23, 24, 26, 27, 32, 33, 34, 37, 38, 40, 41, 43, 45, 47, 48]
        self.assertEqual(len(node_types), total_nodes, f"Total number of nodes should be {total_nodes}")
        self.assertEqual(sum(node_types == True), total_leaf_nodes,
                         f"Total number of leaf nodes should be {total_leaf_nodes}")
        self.assertEqual(sum(node_types == False), total_split_nodes,
                         f"Total number of split nodes should be {total_split_nodes}")
        for lead_node_id in leaf_node_ids:
            self.assertTrue(node_types[lead_node_id], f"Node with id {lead_node_id} should be a leaf node")

    def test_get_leaf_sample_counts(self):
        leaf_id, leaf_samples = ShadowDecTree.get_leaf_sample_counts(self.tree_model)

        leaf_samples_raw = [1, 1, 68, 1, 81, 18, 53, 57, 6, 1, 1, 1, 3, 22, 6, 5, 3, 3, 13, 422, 21, 8, 8, 49, 39]

        for i in range(0, len(leaf_samples)):
            self.assertEqual(leaf_samples[i], leaf_samples_raw[i],
                             f"Leaf with id {leaf_id[i]} should have {leaf_samples_raw[i]} samples")

    def test_get_leaf_sample_counts_by_class(self):
        index, leaf_samples_0, leaf_samples_1 = ShadowDecTree.get_leaf_sample_counts_by_class(self.tree_model)

        leaf_samples_0_raw = [1.0, 0.0, 6.0, 1.0, 0.0, 1.0, 15.0, 27.0, 6.0, 0.0, 0.0, 1.0, 2.0, 21.0, 0.0, 2.0, 3.0,
                              0.0, 11.0, 378.0, 15.0, 0.0, 8.0, 22.0, 29.0]
        leaf_samples_1_raw = [0.0, 1.0, 62.0, 0.0, 81.0, 17.0, 38.0, 30.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 6.0, 3.0, 0.0,
                              3.0, 2.0, 44.0, 6.0, 8.0, 0.0, 27.0, 10.0]

        for i in range(0, len(leaf_samples_0)):
            self.assertEqual(leaf_samples_0[i], leaf_samples_0_raw[i],
                             f"Leaf with id {index[i]} should have {leaf_samples_0_raw[i]} samples")

        for i in range(0, len(leaf_samples_1)):
            self.assertEqual(leaf_samples_1[i], leaf_samples_1_raw[i],
                             f"Leaf with id {index[i]} should have {leaf_samples_1_raw[i]} samples")


if __name__ == "__main__":
    unittest.main()
