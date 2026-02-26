#!/usr/bin/env python3
""" 5-build_decision_tree.py """

import numpy as np


class Node:
    """ Node class """

    def __init__(self,
                 feature=None,
                 threshold=None,
                 left_child=None,
                 right_child=None,
                 is_root=False, depth=0):
        """ __init__ method """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """ max_depth_below method """
        if self.is_leaf:
            return self.depth
        else:
            return max(
                self.left_child.max_depth_below(),
                self.right_child.max_depth_below()
            )

    def count_nodes_below(self, only_leaves=False):
        """ count_nodes_below method """
        left = self.left_child.count_nodes_below(only_leaves=only_leaves)
        right = self.right_child.count_nodes_below(only_leaves=only_leaves)
        if only_leaves:
            return left + right
        else:
            return 1 + left + right

    def __str__(self):
        """ __str__ method """
        if self.is_root:
            prefix = "root"
        else:
            prefix = "-> node"

        result = (f"{prefix} [feature={self.feature}, "
                  f"threshold={self.threshold}]\n")
        result += self.left_child_add_prefix(self.left_child.__str__())
        result += self.right_child_add_prefix(self.right_child.__str__())
        return result

    def left_child_add_prefix(self, text):
        """ left_child_add_prefix method """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x.strip():
                new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """ right_child_add_prefix method """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x.strip():
                new_text += ("       " + x) + "\n"
        return new_text

    def get_leaves_below(self):
        """ get_leaves_below method """
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """ update_bounds_below method """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        for child in [self.left_child, self.right_child]:
            if child is None:
                continue

            child.upper = self.upper.copy()
            child.lower = self.lower.copy()
            if child == self.left_child:
                child.lower[self.feature] = self.threshold
            else:
                child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """ update_indicator method """
        def is_large_enough(x):
            return np.all(
                np.array(
                    [np.greater(x[:, key],
                                self.lower[key])
                     for key in self.lower.keys()]),
                axis=0)

        def is_small_enough(x):
            return np.all(
                np.array(
                    [np.less_equal(x[:, key],
                                   self.upper[key])
                     for key in self.upper.keys()]),
                axis=0)

        self.indicator = lambda x: is_large_enough(x) & is_small_enough(x)


class Leaf(Node):
    """ Leaf class """

    def __init__(self, value, depth=None):
        """ __init__ method """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """ max_depth_below method """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """ count_nodes_below method """
        return 1

    def __str__(self):
        """ __str__ method """
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """ get_leaves_below method """
        return [self]

    def update_bounds_below(self):
        """ update_bounds_below method """
        pass


class Decision_Tree():
    """ Decision_Tree class """

    def __init__(self,
                 max_depth=10,
                 min_pop=1,
                 seed=0,
                 split_criterion="random",
                 root=None
                 ):
        """ __init__ method """
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
        """ depth method """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ count_nodes method """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """ __str__ method """
        return self.root.__str__()

    def get_leaves(self):
        """ get_leaves method """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """ update_bounds method """
        self.root.update_bounds_below()
