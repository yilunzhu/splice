from typing import List, Dict, Tuple
import io
import os
import json
import re
from fnmatch import fnmatch
from operator import itemgetter
from nptree import NPTree


def generate_tree(tree, const_type):
    tree = re.sub(r'(\(\w+?)- ', r'\1 ', tree)  # handle special cases, such as CONJP-
    # tree = re.sub(r'\(XX and\)', r'(CC and)', tree)   # handle special cases, such as (XX and)
    if 'pred' not in const_type:
        tree = remove_additional_tags(tree)
        tree = remove_equal_trace(tree)
    tree = re.sub(r'-\d+', '', tree)
    tree = remove_function_tags(tree)
    tree = tree.replace('\n', '')
    if tree.startswith('( ('):
        tree = re.sub(r'^\( \(', r'(', tree)
        tree = re.sub(r'\)$', '', tree)
    try:
        return NPTree.fromstring(tree)
    except:
        try:
            NPTree.fromstring('(S '+tree+')')
        except:
            return None


def remove_equal_trace(tree: str):
    # tree = re.sub(r'\(NP[\w\W]{0,10} \(-NONE- [\w\W]+?\)\)', '', tree)
    tree = re.sub(r'=\d+', '', tree)
    return tree


def remove_function_tags(tree: str):
    tree = re.sub(r'(\w)-[-\w]+? ', r'\1 ', tree)
    return tree


def reverse_labels(np_list, gold_entities, sent_toks):
    """
    Check if the mention is a NP
    """
    spans = []
    np_spans = [item[1] for item in np_list]
    for gold_span in gold_entities:
        if len(gold_span) != 2:
            continue
        if gold_span not in np_spans:
            spans.append(sent_toks[gold_span[0]:gold_span[1]])
    return spans


def find_np(tree, mapping_dict: Dict, np_list: List) -> List:
    """
    Recursively find NPs from a constituent tree
    :param tree: NLTK tree class
    :param np_list: where to store the NPs found in the tree
    :return: np_list
    """
    if type(tree[0]) == str:
        return np_list
    for child in tree:
        if type(child) == str:
            continue
        if child.label().startswith('NP') or \
                child.label().startswith('NX') or \
                child.label().startswith('NML') or \
                child.label().startswith('PRP'):    # or child.label().startswith('PRP')
            absolute_tree_location_start = child.treeposition() + child.leaf_treeposition(0)
            absolute_tree_location_end = child.treeposition() + child.leaf_treeposition(len(child.leaves())-1)
            flat_idx = (mapping_dict[absolute_tree_location_start], mapping_dict[absolute_tree_location_end]+1)
            np_list.append([child.leaves(), flat_idx, child, child.treeposition()])
        find_np(child, mapping_dict, np_list)
    return np_list


def get_nodes(tree, nodes: List):
    if type(tree[0]) == str:
        nodes.append(tree.label())
        return nodes
    for child in tree:
        nodes.append(tree.label())
        if type(child) != str:
            get_nodes(child, nodes)
    return nodes


def create_labels(np_list: List, gold_entities: List[Tuple]) -> List[int]:
    """
    Check which NPs are entities and which are not
    :param np_list: NPs extracted from constituent trees
    :param gold_entities: gold entities extracted from dependency conllu files
    :return: labels: a label list
    """
    labels = []
    for np in np_list:
        span = np[1]
        if span in gold_entities:
            labels.append(1)
        else:
            labels.append(0)
    return labels


def truncate_and_padding(labels, max_len):
    labels = list(labels)
    truncated = labels[:max_len]
    padded = truncated + [''] * (max_len - len(truncated))
    return padded


def add_tree_info(nps, sent_data):
    for i, np in enumerate(nps):
        feats = [0] * len(sent_data['np_feats'][0])
        if len(np) > 2:
            dist_list = [min(abs(sent_data['np_span'][x][-1] - sent_data['np_span'][i][0]),
                             sent_data['np_span'][x][0] - sent_data['np_span'][i][-1]) for x in np]
            min_dist_idx = dist_list.index(min(dist_list))
            feats = sent_data['np_feats'][np[min_dist_idx]]
        elif np:
            feats = sent_data['np_feats'][np[0]]

        sent_data['feats'].append(list(sent_data['np_feats'][i]) + list(feats))


def get_tree_labels(tree, labels):
    if type(tree[0]) == str:
        return labels
    for child in tree:
        if type(child) == str:
            continue
        labels.append(child.label())
        get_tree_labels(child, labels)
    return labels


def save_data(data, file_dir):
    with io.open(file_dir, 'w', encoding='utf8') as f:
        json.dump(data, f)


def _remove_tag(tree, tag):
    if f'({tag}' not in tree:
        return tree
    del_list = []
    word_level_tag = []
    starts = []
    for m in re.finditer(fr'\({tag} ', tree):
        start_idx = m.start()
        end_idx = m.end()
        starts.append(start_idx)
        if tree[end_idx] != r'(':   # if the tag is a word-level tag
            word_level_tag.append(True)
        else:
            word_level_tag.append(False)
        stack = []
        for id in range(start_idx, len(tree)):
            if tree[id] == '(':
                stack.append('(')
            if tree[id] == ')':
                stack.pop()
            if not stack:
                del_list.append(id)
                break

    indices, sorted_del_list = zip(*sorted(enumerate(del_list), key=itemgetter(1), reverse=True))
    for idx, end_idx in zip(indices, sorted_del_list):
        start_idx = starts[idx]
        is_word_tag = word_level_tag[idx]
        if is_word_tag and tag == 'XX':
            a = 1
            # continue
            # tree = tree[:start_idx] + tree[end_idx + 2:]
        tree = tree[:end_idx] + tree[end_idx + 1:]
        # tree = tree[:start_idx] + tree[start_idx+len(fr'\({tag} ')-1:end_idx]+tree[end_idx+1:]
    tree = re.sub(fr'\({tag} ', '', tree)

    return tree


def remove_additional_tags(tree):
    """
    Remove additional functional tags, such as ADD, EMBED from the tree
    :param tree: raw string
    :return: edited string
    """
    if '(EMBED' in tree:
        a = 1
    # TODO: if a tree has multiple EMBED tags
    # ADD
    tree = _remove_tag(tree, tag='ADD')
    # EMBED
    tree = _remove_tag(tree, tag='EMBED')
    # XX
    # tree = _remove_tag(tree, tag='XX')
    return tree


def parse_filenames(dirname, pattern="*conll"):
    """ Walk a nested directory to get all filename ending in a pattern """
    for path, subdirs, files in os.walk(dirname):
        for name in files:
            if fnmatch(name, pattern):
                yield os.path.join(path, name)


def add_entities(fields, sent_entities, tok_id, count_same_entities):
    for e in fields[-1].split('|'):
        e_id = e.strip(')').strip('(').split('-')[0]

        # nested entities
        if e_id in sent_entities and len(sent_entities[e_id]['boundary']) == 1 and '(' in e and ')' in e:
            sent_entities[f'{e_id}_nested'] = {'boundary': [tok_id, tok_id + 1]}
            # print(1)
            continue
        # if the antecedent exists
        if e_id in sent_entities and len(sent_entities[e_id]['boundary']) == 2:
            prev_mention_len = sent_entities[e_id]['boundary'][1] - sent_entities[e_id]['boundary'][0]
            if tok_id == sent_entities[e_id]['boundary'][1] and prev_mention_len == 1:  # if the last token has the same entity id as the current one
                sent_entities[e_id]['boundary'][1] = tok_id + 1
            else:
                new_e_id = f'{e_id}_{count_same_entities[e_id]}'
                sent_entities[new_e_id] = sent_entities[e_id]
                del sent_entities[e_id]
        if '(' in e and ')' in e:
            sent_entities[e_id] = {'boundary': [tok_id, tok_id + 1]}
            count_same_entities[e_id] += 1
        elif '(' in e:
            sent_entities[e_id] = {'boundary': [tok_id, ]}
        elif ')' in e:
            if e_id not in sent_entities:
                continue
            sent_entities[e_id]['boundary'].append(tok_id + 1)
            count_same_entities[e_id] += 1
    return sent_entities, count_same_entities


def find_largest_span(cur_span):
    np_tree = cur_span
    largest_span = cur_span.span
    while cur_span.parent():
        if cur_span.parent().label() in ['NP', 'NX', 'NML', 'PRP'] and ' '.join(np_tree.leaves()) in ' '.join(np_tree.parent_text):
            largest_span = cur_span.parent().span
        cur_span = cur_span.parent()
    return largest_span


def find_smallest_span(cur_span, smallest_span):
    # smallest_span = cur_span.span
    children_head_labels = [1 if child.label() in ['NP', 'NX', 'NML', 'PRP'] else 0 for child in cur_span.children]
    if sum(children_head_labels) >= 1:
        for child in cur_span.children:
            if child.label() in ['NP', 'NX', 'NML', 'PRP']:
                smallest_span = find_smallest_span(child, child.span)
                break
        return smallest_span
    else:
        return smallest_span