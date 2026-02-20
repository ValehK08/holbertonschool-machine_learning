#!/usr/bin/env python3
""" 0-build_decision_tree.py """
import numpy as np


class Node:
    """ Node """
    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        is_root=False,
        depth=0
    ):
        """ initialize the node """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """ maximum depth below the current Node """
        max_d = self.depth
        if self.left_child:
            max_d = max(max_d, self.left_child.max_depth_below())
        if self.right_child:
            max_d = max(max_d, self.right_child.max_depth_below())
        return max_d


class Leaf(Node):
    """ Leaf """
    def __init__(self, value, depth=None):
        """ initialize the leaf """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """ max_depth_below """
        return self.depth


class Decision_Tree():
    """ Decision Tree """
    def __init__(
        self,
        max_depth=10,
        min_pop=1,
        seed=0,
        split_criterion="random",
        root=None
    ):
        """ initialize a tree """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """ depth """
        return self.root.max_depth_below()
