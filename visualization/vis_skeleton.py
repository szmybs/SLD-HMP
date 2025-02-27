# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import warnings


class VisSkeleton:
    def __init__(self, parents, joints_left, joints_right):
        assert len(joints_left) == len(joints_right)
        
        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_metadata()
        self._compute_adjacency_matrix()
        return
    
    def num_joints(self):
        return len(self._parents)
    
    def parents(self):
        return self._parents
    
    def has_children(self):
        return self._has_children
    
    def children(self):
        return self._children
    
    def remove_joints(self, joints_to_remove):
        """
        Remove the joints specified in 'joints_to_remove'.
        """
        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        for i in range(len(self._parents)):
            while self._parents[i] in joints_to_remove:
                self._parents[i] = self._parents[self._parents[i]]
                
        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)
        
        if self._joints_left is not None:
            new_joints_left = []
            for joint in self._joints_left:
                if joint in valid_joints:
                    new_joints_left.append(joint - index_offsets[joint])
            self._joints_left = new_joints_left
        if self._joints_right is not None:
            new_joints_right = []
            for joint in self._joints_right:
                if joint in valid_joints:
                    new_joints_right.append(joint - index_offsets[joint])
            self._joints_right = new_joints_right
        self._compute_metadata()
        self._compute_adjacency_matrix()
        return valid_joints
    
    
    def adjust_connection_manually(self, adjust_list):
        for ad in adjust_list:
            self._parents[ad[0]] = ad[1]
        self._compute_metadata()
        self._compute_adjacency_matrix()
        return
    
    
    def add_new_nodes_manually(self, new_items:list):
        old_length = self._parents.shape[0]
        new_length = old_length + len(new_items)
        
        new_parents = np.zeros(shape=(new_length,), dtype=int)
        new_parents[:old_length] = self._parents
        for idx, item in enumerate(new_items):
            new_parents[old_length+idx] = item
        self._parents = new_parents
        
        self._compute_metadata()
        self._compute_adjacency_matrix()
        return
    
    
    def joints_left(self):
        return self._joints_left
    
    def joints_right(self):
        return self._joints_right
        
    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)
        return
    
    def _compute_adjacency_matrix(self):
        node_n = len(self._parents)
        adj_matrix = [np.where(self._parents==i, 1, 0) for i in range(node_n)]
        for i, adm in enumerate(adj_matrix):
            if self._parents[i] != -1:
                adm[self._parents[i]] = 1
        self.adj_matrix = np.array(adj_matrix)
        # print(self.adj_matrix.tolist())
        self.adj_matrix_T = np.transpose(self.adj_matrix)
        return
    

    def compute_bones_length(self, cartesian:np.array):
        warnings.warn("The function(compute_bones_length) seems wrong.", DeprecationWarning)
        mask = self.adj_matrix - np.diag(cartesian.size())
        c1 = np.tile(cartesian, (cartesian.size(), 1))
        c2 = np.transpose(c1)
        sub = np.multiply(np.subtract(c1, c2), mask)
        squ = np.power(sub, 2)
        bones_length_matrix = np.mean(squ, -1)
        return bones_length_matrix
    