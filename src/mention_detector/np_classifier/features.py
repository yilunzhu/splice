from collections import defaultdict
from utils import truncate_and_padding

label2idx = {'': 0, 'S': 1, 'PP': 2, 'IN': 3, 'NP': 4, 'NN': 5, 'PRP$': 6, 'ADJP': 7, 'RB': 8, 'JJ': 9, ',': 10,
             'SBAR-NOM': 11, 'WHADVP': 12, 'WRB': 13, 'NP-SBJ': 14, 'VP': 15, 'VBZ': 16, '-LRB-': 17, 'NNS': 18,
             'NP-TMP': 19, 'NNP': 20, 'CD': 21, '-RRB-': 22, ':': 23, 'PRP': 24, 'VBP': 25, 'PP-LOC-PRD': 26, 'DT': 27,
             '.': 28, 'VBD': 29, 'VBN': 30, 'PRT': 31, 'RP': 32, 'PP-TMP': 33, 'CC': 34, 'SBAR-TMP': 35,
             'PP-LOC-CLR': 36, 'NP-LGS': 37, '``': 38, "''": 39, 'NML': 40, 'PP-PRD': 41, 'ADVP': 42, 'SBAR': 43,
             'SBAR-NOM-SBJ': 44, 'WHNP': 45, 'WP': 46, 'RBS': 47, 'SBAR-PRD': 48, 'EX': 49, 'TO': 50, 'VB': 51,
             'NP-PRD': 52, 'ADJP-PRD': 53, 'SBAR-ADV': 54, 'VBG': 55, 'POS': 56, 'S-ADV': 57, 'PP-LOC': 58,
             'PP-CLR': 59, 'SQ': 60, 'S-NOM-SBJ': 61, 'JJS': 62, 'FRAG': 63, 'QP': 64, 'JJR': 65, 'NML-TTL': 66,
             'ADVP-TMP': 67, 'S-NOM': 68, 'WDT': 69, 'MD': 70, 'PP-DIR': 71, 'ADVP-MNR': 72, 'SINV': 73, 'S-TPC': 74,
             'PRN': 75, 'HYPH': 76, 'ADVP-CLR': 77, '$': 78, 'NP-ADV': 79, 'PP-DTV': 80, 'PP-PRP': 81, 'NP-CLR': 82,
             'NP-TTL': 83, 'NNPS': 84, 'S-PRP': 85, 'ADVP-LOC': 86, 'S-CLR': 87, 'ADJP-LOC': 88, 'RBR': 89, 'SYM': 90,
             'NP-LOC': 91, 'UCP': 92, 'VP-TPC': 93, 'PP-MNR': 94, 'ADVP-LOC-PRD': 95, 'S-PRD': 96, 'NP-VOC': 97,
             'S-TTL': 98, 'X-ADV': 99, 'NAC': 100, 'PDT': 101, 'ADVP-PRD': 102, 'SBAR-PRP': 103, 'ADVP-PRD-TPC': 104,
             'PP-EXT': 105, 'S-HLN': 106, 'CONJP': 107, 'SBAR-NOM-PRD': 108, 'NP-HLN': 109, 'ADVP-DIR': 110,
             'PP-TMP-CLR': 111, 'WHPP': 112, 'NP-EXT': 113, 'ADVP-LOC-PRD-TPC': 114, 'PP-PUT': 115, 'WP$': 116,
             'RRC': 117, 'SBARQ': 118, 'NP-MNR': 119, 'FW': 120, 'PP-CLR-LOC': 121, 'INTJ': 122, 'UH': 123,
             'PP-DIR-CLR': 124, 'ADJP-CLR': 125, 'ADJP-TPC': 126, 'UCP-MNR': 127, 'NP-TTL-PRD': 128, 'PP-TTL': 129,
             'NP-TTL-SBJ': 130, 'SBAR-PRP-PRD': 131, 'UCP-PRD': 132, 'S-CLF-TPC': 133, 'S-CLF': 134, 'S-MNR': 135,
             'PP-LOC-PRD-TPC': 136, 'NP-TMP-HLN': 137, 'ADVP-PRP': 138, 'SBAR-CLR': 139, 'X-EXT': 140, 'WHADJP': 141,
             'SBAR-MNR': 142, 'PP-TPC': 143, 'SBAR-LOC': 144, 'X': 145, 'NFP': 146, 'ADVP-LOC-CLR': 147, 'S-TMP': 148,
             'PP-BNF': 149, 'S-NOM-PRD': 150, 'ADJP-PRD-TPC': 151, 'S-SBJ': 152, 'LST': 153, 'LS': 154, 'SBAR-SBJ': 155,
             'UCP-PRP': 156, 'S-NOM-LGS': 157, 'UCP-LOC': 158, 'FRAG-ADV': 159, 'NP-TPC': 160, 'PP-LOC-TPC': 161,
             'S-PRP-CLR': 162, 'UCP-TMP': 163, 'ADVP-PRD-LOC': 164, 'PP-PRD-LOC': 165, 'SBARQ-PRD': 166,
             'FRAG-TPC': 167, 'NP-BNF': 168, 'ADVP-TMP-TPC': 169,
             # tags in GUM
             'NX': 170, 'VP-SBJ': 171, 'NX-SBJ': 172, 'NX-PRD': 173, 'VP-PRP': 174, 'ADVP-EXT': 175, 'NP-TMP-CLR': 176,
             'ADJP-ADV': 177,
             # tags in ontonotes
             'META': 178, 'AFX': 179, 'XX': 180, '_SP': 181, 'ADD': 182, '#': 183,
             'ADJ': 9,
             'ROOT': 184}


phrase_tags2idx = {'S': 0, 'SBAR': 1, 'SBARQ': 2, 'SINV': 3, 'SQ': 4, 'ADJP': 5, 'ADVP': 6, 'CONJP': 7, 'FRAG': 8,
               'INTJ': 9, 'LST': 10, 'NAC': 11, 'NP': 12, 'NX': 13, 'PP': 14, 'PRN': 15, 'PRT': 16, 'QP': 17,
               'RRC': 18, 'UCP': 19, 'VP': 20, 'WHADJP': 21, 'WHAVP': 22, 'WHNP': 23, 'WHPP': 24, 'X': 25, 'CC': 26,
               'CD': 27, 'DT': 28, 'EX': 29, 'FW': 30, 'IN': 31, 'JJ': 32, 'JJR': 33, 'JJS': 34, 'LS': 35, 'MD': 36,
               'NN': 37, 'NNS': 38, 'NNP': 39, 'NNPS': 40, 'PDT': 41, 'POS': 42, 'PRP': 43, 'PRP$': 44, 'RB': 45,
               'RBR': 46, 'RBS': 47, 'RP': 48, 'SYM': 49, 'TO': 50, 'UH': 51, 'VB': 52, 'VBD': 53, 'VBG': 54, 'VBN': 55,
               'VBP': 56, 'VBZ': 57, 'WDT': 58, 'WP': 59, 'WP$': 60, 'WRB': 61, 'META': 62, 'AFX': 63,

               'ROOT': 62}
form_tags = {'': 0, 'ADV': 1, 'NORM': 2}
grammar_tags = {'': 0, 'DTV': 1, 'LGS': 2, 'PRD': 3, 'PUT': 4, 'SBJ': 5, 'TPC': 6}
adv_tags = {'': 0, 'BNF': 1, 'DIR': 2, 'EXT': 3, 'LOC': 4, 'MNR': 5, 'PRP': 6, 'TMP': 7}
misc_tags = {'': 0, 'CLR': 1, 'CLF': 2, 'HLN': 3, 'TTL': 4}
preptoken2idx = {'': 0, 'in': 1, 'of': 2, 'from': 3, 'on': 4, 'at': 5, 'about': 6, 'between': 7, 'upon': 8, 'into': 9,
                 'to': 10, 'as': 11, 'through': 12, 'by': 13, 'for': 14, 'within': 15, 'with': 16, 'despite': 17,
                 'like': 18, 'across': 19, 'without': 20, 'around': 21, 'over': 22, 'among': 23, 'including': 24,
                 'since': 25, 'besides': 26, 'than': 27, 'after': 28, 'before': 29, 'that': 30, 'under': 31, 'above': 32,
                 'during': 33, 'beyond': 34, 'behind': 35, 'per': 36, 'below': 37, 'until': 38, 'up': 39,
                 'throughout': 40, 'while': 41, 'onto': 42, 'beside': 43, 'near': 44, 'against': 45, 'except': 46,
                 'out': 47, 'til': 48, 'vs': 49, 'via': 50, 'along': 51, 'whether': 52, 'unlike': 53, 'if': 54,
                 'off': 55, 'towards': 56, 'opposite': 57, 'alongside': 58, 'outside': 59, 'aboard': 60, 'round': 61,
                 'inside': 62, 'past': 63, 'but': 64, 'vs.': 65, 'amongst': 66, 'down': 67, 'amidst': 68, 'underneath': 69,
                 'minus': 70, 'astride': 71}
definite_markers = ['i', 'me', 'my', 'you', 'your', 'we', 'our', 'they', 'their', 'the', 'a', 'an', 'he', 'his', 'she', 'her',
                    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'here', 'there',
                    'many', 'most', 'any', 'all', 'no', 'it', 'its', 'other', 'these', 'this', 'that', 'those']
on2gum = {'TOP': 'ROOT'}


def is_feat_exist(labels, trigger):
    value = 0.0
    if not labels:
        return value
    for l in labels:
        if l.startswith(trigger):
            value = 1.0
    return value


def len_in_labels(labels, trigger):
    return float(len([l for l in labels if l.startswith(trigger)]))


def find_copula(tree):
    """
    Loop the left siblings of the NP and see if copula is there (In OntoNotes, NP after copula are not mentions)
    :param tree: NP tree
    :return: whether copula in siblings or not
    """
    siblings_cop = 0.0
    left_sibs = tree.left_siblings
    left_sib_labels = tree.left_siblings_labels
    if 'CC' in left_sib_labels or 'CONJP' in left_sib_labels:   # if the NP is in a conjunction, check its parent's sibling
        left_sibs = tree.parent().left_siblings
    if left_sibs:
        for l in left_sibs:
            if 'VB' in l.label() and l.lemma[0] == 'be':
                siblings_cop = 1.0

    return siblings_cop


def check_span_include(i, j):
    if i[0] <= j[0] and j[1] <= i[1]:
        return True
    return False


def check_span_include_reverse(i, j):
    if j[0] <= i[0] and i[1] <= j[1]:
        return True
    return False


def check_sublist(sublist, lst):
    """
    Check if span A is under than span B
    :param sublist: span A
    :param lst: span B
    """
    for idx in range(len(lst) - len(sublist) + 1):
        if lst[idx: idx + len(sublist)] == sublist:
            return True
    return False


def find_largest_span(span_rels_reverse, span):
    if span not in span_rels_reverse:
        return span
    elif span == span_rels_reverse[span]:
        return span
    else:
        return find_largest_span(span_rels_reverse, span_rels_reverse[span])


def level_of_nps(all_spans, all_pos):
    span_rels, span_rels_reverse = defaultdict(list), {}

    # find which np span is nested into which np span
    # it will list all children np for the np span
    for i in range(len(all_spans)):
        for j in range(i+1, len(all_spans)):
            if check_sublist(list(all_pos[i]), list(all_pos[j])):
                span_rels[all_spans[i]].append(all_spans[j])
                span_rels_reverse[all_spans[j]] = all_spans[i]

    largest_span_rels_reverse = {}  # find the largest span covering the current span
    top_spans, leaf_spans = [], []  # find all spans that are spans which do not have np parents and which do not have np children
    for span in all_spans:
        if span not in span_rels_reverse:
            top_spans.append(span)
        if span not in span_rels:
            leaf_spans.append(span)
        # find the largest span recursively
        largest_span_rels_reverse[span] = find_largest_span(span_rels_reverse, span)

    # assign a level to each np span
    levels = {}
    level = 0
    while top_spans:
        new_stack = []
        for span in top_spans:
            levels[span] = level

            for child_span in span_rels[span]:
                if span_rels_reverse[child_span] == span and child_span not in new_stack:
                    new_stack.append(child_span)
        top_spans = new_stack
        level += 1

    return levels, span_rels, span_rels_reverse, largest_span_rels_reverse


def select_np_features(np, np_idx, sent_data):
    # define
    nums_np = len(sent_data['np'])
    nums_parent_np = 0.0
    nums_sibling_np = 0.0
    nums_child_np = 0.0

    np_span = sent_data['np_span'][np_idx]
    parent_span = sent_data['parent_span'][np_idx]
    children_labels = np.children_labels
    # largest_span = sent_data['np_span'][np_idx]
    # smallest_span = sent_data['np_span'][np_idx]

    np_parent, np_sibling, np_child = [], [], []
    for i in range(nums_np):
        if i == np_idx:
            continue
        cur_np_start, cur_np_end = sent_data['np_span'][i]
        cur_parent_span = sent_data['parent_span'][i]
        # 1. If there is a larger NP span scoping over the current NP
        if cur_np_start <= np_span[0] and np_span[1] <= cur_np_end:
            nums_parent_np += 1
            np_parent.append(i)
            # # find the largest np span
            # if cur_np_start <= largest_span[0] and largest_span[1] <= cur_np_end:
            #     largest_span = sent_data['np_span'][i]
        # 2. If there is a NP span that has the same parent span as the current NP
        if (cur_np_end < np_span[0] or np_span[1] < cur_np_start) and parent_span == cur_parent_span:
            nums_sibling_np += 1
            np_sibling.append(i)
        # 3. If there is a nested NP span within the current NP
        if np_span[0] <= cur_np_start and cur_np_end <= np_span[1]:
            nums_child_np += 1
            np_child.append(i)
            # # find the smallest np span
            # if smallest_span[0] <= cur_np_start and cur_np_end <= smallest_span[1]:
            #     smallest_span = sent_data['np_span'][i]

    # token-level features
    np_tokens = sent_data['sent'][np_span[0]:np_span[1]]

    if children_labels:
        # POS of the first token
        label = np.children[0].label().split('=')[0]
        first_pos = float(label2idx[label])
        # def_np = 1.0 if np_tokens[0].lower() in definite_markers or children_labels[0].startswith('NNP') else 0.0
        def_np = 0.0 if children_labels[0].startswith('NN') else 1.0
    else:
        # POS of the first token
        label = np.label().split('=')[0]
        first_pos = float(label2idx[label])
        # def_np = 1.0 if np_tokens[0].lower() in definite_markers or children_labels[0].startswith('NNP') else 0.0
        def_np = 0.0 if np.label().startswith('NN') else 1.0
    possessive = 1.0 if "'s" in np_tokens else 0.0
    left_capitalized_prep = 1.0 if sent_data['np_span'][np_idx][0] > 0 and sent_data['sent'][sent_data['np_span'][np_idx][0]-1][0].isupper() else 0.0

    # check if the NP is under a big SBAR of a NP (if it is, the VP and other XP can also be part of a mention)
    if_clause = 0.0
    while np:
        if np.parent():
            if np.label().startswith('SBAR') and np.parent().label().startswith('NP'):
                if_clause = 1.0
                break
            np = np.parent()
        else:
            break

    # check if the NP is a proper noun
    if_nnp = 0.0
    if children_labels:
        NNP_FLAG = True
        for l in children_labels:
            if not l.startswith('NNP'):
                NNP_FLAG = False
        if NNP_FLAG:
            if_nnp = 1.0

    np_feats = [nums_parent_np, nums_sibling_np, nums_child_np, first_pos, def_np, possessive, if_clause, left_capitalized_prep, if_nnp]
    span_rels = (np_parent, np_sibling, np_child)
    return np_feats, span_rels


def select_parent_features(np):
    parent_label = np.parent().label() if np.parent() else ''
    parent_label_id = float(label2idx[parent_label]) if parent_label in label2idx else float(label2idx[on2gum[parent_label]])
    return [parent_label_id]


def check_prep(parent, sibling):
    prep = 0.0
    try:
        if parent.label() == 'PP' and sibling.label() == 'IN':
            prep = parent.leaves()[0].lower()
            prep = float(preptoken2idx[prep])
    except:
        prep = 0.0
    return prep


def select_siblings_features(np, max_len=3):
    sibling_labels = np.sibling_labels
    left_siblings = np.left_siblings_labels
    right_siblings = np.right_siblings_labels

    # if the siblings are [, NP], it's likely an appositive and the outer span of appositives should not be removed
    if_appos = 0.0
    if not left_siblings:
        if_appos = 1.0
        for label in right_siblings:
            if label not in [',', 'NP']:
                if_appos = 0.0

    left_siblings = truncate_and_padding(set(left_siblings), max_len)
    left_siblings_id = [float(label2idx[l]) if l in label2idx else float(label2idx[on2gum[l]]) for l in left_siblings]
    right_siblings = truncate_and_padding(set(right_siblings), max_len)
    right_siblings_id = [float(label2idx[l]) if l in label2idx else float(label2idx[on2gum[l]]) for l in right_siblings]

    # if the head is a PP, the token of the left & right sibling
    left_prep = check_prep(np.parent(), np.left_sibling())
    right_prep = check_prep(np.parent(), np.right_sibling())

    # if the siblings have copula
    siblings_cop = find_copula(np)

    siblings_feats = left_siblings_id + right_siblings_id + [siblings_cop, left_prep, right_prep, if_appos]
    return siblings_feats


def select_children_features(children_labels, max_len=3):
    # select the first max_len labels
    first_children_labels = truncate_and_padding(children_labels, max_len)
    first_children_labels_id = [float(label2idx[l]) for l in first_children_labels]

    # select the last max_len labels
    last_children_labels = truncate_and_padding(children_labels[::-1], max_len)
    last_children_labels_id = [float(label2idx[l]) for l in last_children_labels[::-1]]

    return first_children_labels_id+last_children_labels_id


def select_node_features(np, np_idx, sent_data):
    np_feats, span_rels = select_np_features(np, np_idx, sent_data)
    sib_feats = select_siblings_features(np)
    parent_feats = select_parent_features(np)
    children_feats = select_children_features(np.children_labels)

    feats = np_feats + sib_feats + parent_feats + children_feats
    return feats, span_rels


def select_span_features(np, np_idx, sent_data, levels, level_rels, level_rels_reverse, largest_levels_rels_reverse):
    # current node features
    node_feats, span_rels = select_node_features(np, np_idx, sent_data)

    # parent's features
    parent_node_features, _ = select_node_features(np.parent(), np_idx, sent_data)

    np_height = np.height()
    np_len = len(sent_data['np_span'][np_idx])
    np_interact_feats = select_np_interact_features(sent_data['np_span'][np_idx], sent_data['pos'], levels, level_rels, level_rels_reverse, largest_levels_rels_reverse)
    feats = node_feats + parent_node_features + [np_height, np_len] + np_interact_feats
    return feats, span_rels


def select_np_interact_features(span, sent_pos, levels, level_rels, level_rels_reverse, largest_levels_rels_reverse):
    # number of np spans on the left and on the right with the largest np span
    rest_spans = [i for i in level_rels[largest_levels_rels_reverse[span]] if i != span]
    left_spans = [i for i in rest_spans if i[1] < span[0]]
    right_spans = [i for i in rest_spans if span[1] < i[0]]
    largest_parent_span = largest_levels_rels_reverse[span]
    children_spans = level_rels[span]

    if_largest = 1.0 if largest_levels_rels_reverse[span] == span else 0.0
    if_smallest = 1.0 if not level_rels[span] else 0.0

    if_left = 1.0 if left_spans else 0.0
    if_right = 1.0 if right_spans else 0.0

    np_interact_feats = [if_largest, if_smallest, if_left, if_right]

    return np_interact_feats


def get_sibling_token(spans, nps, sent_data):
    left_sib, right_sib = 0.0, 0.0
    left_sib_id, right_sib_id = 0.0, 0.0
    if spans:
        # get left and right siblings
        if sent_data['np_tree'][nps[0]].left_sibling():
            left_sib = float(label2idx[sent_data['np_tree'][nps[0]].left_sibling().label()])
            left_sib_token = sent_data['sent'][spans[0][0]-1]
            left_sib_id = float(preptoken2idx[left_sib_token]) if left_sib_token in preptoken2idx else 0.0
        if sent_data['np_tree'][nps[0]].right_sibling():
            right_sib = float(label2idx[sent_data['np_tree'][nps[0]].right_sibling().label()])
            right_sib_token = sent_data['sent'][spans[0][1]]
            right_sib_id = float(preptoken2idx[right_sib_token]) if right_sib_token in preptoken2idx else 0.0
    feats = [left_sib, left_sib_id, right_sib, right_sib_id]
    return feats


def select_feats(np, np_idx, sent_data, levels, level_rels, level_rels_reverse, largest_levels_rels_reverse, max_len=4):
    """
    Features are used for classification:
        1. base features
            - NP features
                - number of NP in parent
                - number of NP in siblings
                - number of NP in children
            - parent features
            - siblings features
            - children features
        2. extra features
            - parent's siblings features
            - siblings' children features
    """

    # current NP span features
    cur_np_feats, cur_np_rels = select_span_features(np, np_idx, sent_data, levels, level_rels, level_rels_reverse, largest_levels_rels_reverse)

    feats_len = len(cur_np_feats)

    # interaction with other NP spans
    """
    - if it's the largest or smallest span
    - the level of current NP in the NP spans
    """

    parent_nps, sibling_nps, children_nps = cur_np_rels
    np_span = sent_data['np_span'][np_idx]

    # children np
    children_spans = [sent_data['np_span'][cnp] for cnp in children_nps]
    children_np_feats = get_sibling_token(children_spans, children_nps, sent_data)

    return cur_np_feats + children_np_feats
