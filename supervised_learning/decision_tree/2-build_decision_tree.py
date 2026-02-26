#!/usr/bin/env python3
""" 2-build_decision_tree.py """
import numpy as np


class Node:
    """ Node Class """

    def __init__(self,
                 feature=None,
                 threshold=None,
                 left_child=None,
                 right_child=None,
                 is_root=False, depth=0):
        """ Init Method"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """ max_depth_below """
        if self.is_leaf:
            return self.depth
        else:
            return max(
                self.left_child.max_depth_below(),
                self.right_child.max_depth_below()
            )

    def count_nodes_below(self, only_leaves=False):
        """ count_nodes_below """
        left = self.left_child.count_nodes_below(only_leaves=only_leaves)
        right = self.right_child.count_nodes_below(only_leaves=only_leaves)
        if only_leaves:
            return left + right
        else:
            return 1 + left + right

    def __str__(self):
        """ __str__ """
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
        """ left_child_add_prefix """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x.strip():
                new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """ right_child_add_prefix """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x.strip():
                new_text += ("       " + x) + "\n"
        return (new_text)


class Leaf(Node):
    """ Leaf Node """

    def __init__(self, value, depth=None):
        """ init method """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """ max_depth_below """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """ count_nodes_below """
        return 1

    def __str__(self):
        """ __str__ """
        return (f"-> leaf [value={self.value}]")


class Decision_Tree():
    """ Decision Tree Class"""

    def __init__(self,
                 max_depth=10,
                 min_pop=1,
                 seed=0,
                 split_criterion="random",
                 root=None
                 ):
        """ init method """
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

    def count_nodes(self, only_leaves=False):
        """ count nodes """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """ __str__ """
        return self.root.__str__()
