import re
from typing import List, Dict, Tuple
from nltk.tree import Tree, ParentedTree
from copy import deepcopy


class NPTree(ParentedTree):
    def __init__(self, node, children):
        super().__init__(node, children)
        self.tok = []
        self.lemma = []
        self.pos = []
        self.siblings = []
        self.sibling_labels = []
        self.sibling_lemma = []
        self.children = []
        self.children_labels = []
        self.children_lemma = []
        self.left_siblings = []
        self.left_siblings_text = []
        self.left_siblings_span = [-1, -1]
        self.left_siblings_labels = []
        self.right_siblings = []
        self.right_siblings_text = []
        self.right_siblings_span = [-1, -1]
        self.right_siblings_labels = []
        self.parent_text = []
        self.parent_span = ()
        self.span = ()
        self.content_leaves = []

    def add_info(self, sent_tok_info, mapping_dict):
        # get token-level features
        self._add_token_info(mapping_dict, sent_tok_info)

        for subtree in self.subtrees():
            position = subtree.treeposition()

            # get siblings
            self._get_siblings(subtree, position)
            self._get_parent_text(subtree, position, mapping_dict)
            self._get_left_siblings(subtree, position, mapping_dict)
            self._get_right_siblings(subtree, position, mapping_dict)

            # get children
            self._get_children(subtree, position)
        a = 1

    def _add_token_info(self, mapping_dict, sent_tok_info):
        for subtree in self.subtrees():
            position = subtree.treeposition()
            token_idx = sorted([mapping_dict[p] for p in mapping_dict.keys() if position == p[:len(position)]])
            self[position].tok = [sent_tok_info[idx][0] for idx in token_idx]
            self[position].lemma = [sent_tok_info[idx][1] for idx in token_idx]
            self[position].pos = [sent_tok_info[idx][2] for idx in token_idx]
            self[position].span = (token_idx[0], token_idx[-1]+1)

    def _get_siblings(self, subtree, position):
        if subtree.parent():
            for n in subtree.parent():
                if n == subtree or type(n) == str:
                    continue
                # label = self.remove_trace(n.label())
                label = n.label()
                self[position].siblings.append(n)
                self[position].sibling_labels.append(label.split('=')[0])
                self[position].sibling_lemma.extend(n.lemma)

    def _get_left_siblings(self, subtree, position, mapping_dict):
        if subtree.parent():
            for n in subtree.parent():
                if type(n) == str:
                    continue
                if n == subtree:
                    break
                self[position].left_siblings.append(n)
                self[position].left_siblings_text += n.leaves()
                # label = self.remove_trace(n.label())
                label = n.label()
                self[position].left_siblings_labels.append(label.split('=')[0])

                span_start = mapping_dict[n.treeposition() + n.leaf_treeposition(0)]
                span_end = mapping_dict[n.treeposition() + n.leaf_treeposition(len(n.leaves()) - 1)] + 1
                if self[position].left_siblings_span != [-1, -1]:
                    if span_start < self[position].left_siblings_span[0]:
                        self[position].left_siblings_span[0] = span_start
                    if span_end > self[position].left_siblings_span[1]:
                        self[position].left_siblings_span[1] = span_end
                else:
                    self[position].left_siblings_span = [span_start, span_end]

    def _get_right_siblings(self, subtree, position, mapping_dict):
        if subtree.parent():
            for n in subtree.parent():
                if type(n) == str:
                    continue
                if n == subtree or n in self[position].left_siblings:
                    continue
                self[position].right_siblings.append(n)
                self[position].right_siblings_text += n.leaves()
                # label = self.remove_trace(n.label())
                label = n.label()
                self[position].right_siblings_labels.append(label.split('=')[0])

                span_start = mapping_dict[n.treeposition() + n.leaf_treeposition(0)]
                span_end = mapping_dict[n.treeposition() + n.leaf_treeposition(len(n.leaves()) - 1)] + 1
                if self[position].right_siblings_span != [-1, -1]:
                    if span_start < self[position].right_siblings_span[0]:
                        self[position].right_siblings_span[0] = span_start
                    if span_end > self[position].right_siblings_span[1]:
                        self[position].right_siblings_span[1] = span_end
                else:
                    self[position].right_siblings_span = [span_start, span_end]

    def _get_children(self, subtree, position):
        for n in subtree:
            if type(n) == str:
                continue
            # label = self.remove_trace(n.label())
            label = n.label()
            self[position].children.append(n)
            self[position].children_labels.append(label.split('=')[0])
            self[position].children_lemma.extend(self[position].lemma)

    def _get_parent_text(self, subtree, position, mapping_dict):
        if subtree.parent():
            for n in subtree.parent():
                if type(n) == str:
                    continue
                self[position].parent_text = subtree.parent().leaves()
                span_start = mapping_dict[subtree.parent().treeposition() + subtree.parent().leaf_treeposition(0)]
                span_end = mapping_dict[subtree.parent().treeposition() + subtree.parent().leaf_treeposition(len(subtree.parent().leaves())-1)] + 1
                self[position].parent_span = (span_start, span_end)

    def idx_in_tree(self) -> Dict:
        """
        Get the mapping from token indices to tree locations
        """
        mapping_dict = {}
        for idx, tok in enumerate(self.leaves()):
            tree_location = self.leaf_treeposition(idx)
            mapping_dict[tree_location] = idx
        return mapping_dict

    def remove_trace(self):
        deleted_idx = []
        for subtree in self.subtrees():
            for n, child in enumerate(subtree):
                if isinstance(child, str):
                    continue
                if len(list(child.subtrees(filter=lambda x:x.label()=='-NONE-'))) == len(child.leaves()):
                    # Direct deletion will affect the following sisters (if a subtree has two phrases including traces)
                    deleted_idx.append((subtree.treeposition(), n))
                    # del self[subtree.treeposition()][n]
        # Delete the trace from bottom right to top left to avoid wrong tree positions
        for tree_pos, idx in deleted_idx[::-1]:
            del self[tree_pos][idx]

    def remove_edited_tokens(self):
        removed_tags = ['EDITED', ]
        deleted_idx = []
        for subtree in self.subtrees():
            for n, child in enumerate(subtree):
                if isinstance(child, str):
                    continue
                if child.label() in removed_tags:
                    deleted_idx.append((subtree.treeposition(), n))
        for tree_pos, idx in deleted_idx[::-1]:
            del self[tree_pos][idx]

        # the deletion may create empty phrases, like (NP )
        deleted_idx = []
        for subtree in self.subtrees():
            for n, child in enumerate(subtree):
                if isinstance(child, str):
                    continue
                if len(child) == 0:
                    deleted_idx.append((subtree.treeposition(), n))
        for tree_pos, idx in deleted_idx[::-1]:
            del self[tree_pos][idx]
