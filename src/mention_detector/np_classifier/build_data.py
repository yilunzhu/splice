import argparse
from tqdm import tqdm
from features import *
from utils import *
from nltk.stem import SnowballStemmer
import spacy

MAX_NODE_LEN = 4
MAX_NP_LEN = 16
node_label_freq = defaultdict(int)
num_children = []
mention_not_np = []
stemmer = SnowballStemmer('english')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) # just keep tagger for lemmatization

genres = defaultdict(int)


def read_ontonotes_conll(sent: str):
    sent_entities = {}
    sent_text = []
    tok_info = []
    count_same_entities = defaultdict(int)
    for line in sent.split('\n'):
        if line.startswith('#'):
            continue
        fields = line.split()
        tok_id = int(fields[2])
        tok, pos = fields[3], fields[4]

        lemma = stemmer.stem(tok)
        tok_info.append([tok, lemma, pos])
        sent_text.append(tok)
        if '-' in fields[-1]:
            continue
        sent_entities, count_same_entities = add_entities(fields, sent_entities, tok_id, count_same_entities)

    entities = [tuple(fields['boundary']) for fields in sent_entities.values()]
    return sent_text, tok_info, entities


def read_conllu(sent: str):
    sent_entities = {}
    sent_text = []
    tok_info = []
    sent_id = ''
    count_same_entities = defaultdict(int)
    for line in sent.split('\n'):
        if line.startswith('# '):
            if line.startswith('# sent_id'):
                sent_id = line.split('=')[-1].split('-')[-1].strip()
            continue
        fields = line.split('\t')
        if '-' in fields[0]:
            continue
        if '.' in fields[0]:
            continue
        tok_id = int(fields[0]) - 1
        tok, lemma, pos = fields[1], fields[2], fields[4]
        tok_info.append([tok, lemma, pos])
        sent_text.append(tok)
        if 'Entity=' not in fields[-1]:
            continue
        # sent_entities, count_same_entities = add_entities(fields, sent_entities, tok_id, count_same_entities)
        es = [f for f in fields[-1].split('|') if 'Entity=' in f]
        es = es[0].strip('Entity=').split('|')[0].replace(')(', ')|(').replace(')', ')|').replace('(', '|(').replace('||', '|').replace('||', '|').strip('|')
        fields[-1] = es
        sent_entities, count_same_entities = add_entities(fields, sent_entities, tok_id, count_same_entities)

    entities = [tuple(fields['boundary']) for fields in sent_entities.values()]
    return sent_text, tok_info, entities, int(sent_id)


def find_children(trees, np_tree):
    all_children = []
    for tree in trees:
        if tree == np_tree:
            continue
        if type(tree) == str:
            continue
        if type(tree[0]) == str:
            children_labels = [tree.label()]
        else:
            children_labels = [node.label() for node in tree if type(node) != str]
        children_labels = truncate_and_padding(children_labels, MAX_NODE_LEN)
        all_children.append(children_labels)
    all_children = truncate_and_padding(all_children, MAX_NODE_LEN)
    return all_children


def get_children_span(np_tree, max_len=5):
    span_indices = [c.span for c in np_tree.children]
    num_children.append(len(span_indices))

    # padding
    span_indices += [(-1, -1)] * (max_len - len(span_indices))
    span_indices = span_indices[:max_len]
    return span_indices


def read_const_file_only(tree_dir: str, const_type: str) -> List[Dict]:
    filename = tree_dir.split('/')[-1].split('.')[0]
    data_dict = []
    const_sents = io.open(tree_dir, encoding='utf8').read().strip().split('\n\n')

    for sent_id, t in enumerate(const_sents):
        real_sent_id = sent_id + 1
        tree = generate_tree(t)
        tree.remove_trace()
        tree.remove_edited_tokens()

        if not tree:
            continue

        flat_idx_mapping_tree = tree.idx_in_tree()
        np_list = find_np(tree, flat_idx_mapping_tree, [])

        tokens = tree.leaves()
        sent_tok_info = []

        for subtree in tree.subtrees():
            if type(subtree[0]) == str:
                tok = subtree[0]
                lemma = stemmer.stem(tok)
                sent_tok_info.append([tok, lemma, subtree.label()])
        tree.add_info(sent_tok_info, flat_idx_mapping_tree)

        sent_data = {'id': f'{filename}-{real_sent_id}',
                     'sent': tree.leaves(),
                     'tree': str(tree),
                     'pos': tree.pos,
                     'feats': [],
                     'np_feats': [],
                     'np': [],
                     'np_tag': [],
                     'np_tree': [],
                     'np_span': [],
                     'parent_span': [],
                     'left_siblings_span': [],
                     'right_siblings_span': [],
                     'largest_np_span': [],
                     'smallest_np_span': [],
                     'children_span': [],
                     'nodes': [],
                     'tree_pos': [],
                     'labels': [],
                     }

        for i, np in enumerate(np_list):
            # get span indices of the direct children of the current np span
            children_span = get_children_span(np[2])
            sent_data['children_span'].append(children_span)

            sent_data['np'].append(np_list[i][0])
            sent_data['np_tree'].append(np[2])
            sent_data['np_span'].append(np_list[i][1])
            sent_data['np_tag'].append(np[2].label())
            sent_data['parent_span'].append(np_list[i][2].parent_span)
            sent_data['tree_pos'].append(np_list[i][3])
            sent_data['left_siblings_span'].append(np_list[i][2].left_siblings_span)
            sent_data['right_siblings_span'].append(np_list[i][2].right_siblings_span)
            sent_data['nodes'].append(get_nodes(np[2], []))

            # Find the largest np span over the current np
            sent_data['largest_np_span'].append(find_largest_span(np[2]))

            # Find the smallest np span under the current np
            sent_data['smallest_np_span'].append(find_smallest_span(np[2], np[2].span))

        levels, levels_rels, levels_rels_reverse, largest_levels_rels_reverse = level_of_nps(sent_data['np_span'], sent_data['tree_pos'])

        for i, np in enumerate(np_list):
            feats = select_feats(np[2], i, sent_data, levels, levels_rels, levels_rels_reverse, largest_levels_rels_reverse)
            sent_data['feats'].append(feats)

        data_dict.append(sent_data)

    return data_dict


def read_file(tree_dir: str, dep_dir: str, dataset: str, const_type: str) -> List[Dict]:
    filename = dep_dir.split('/')[-1].split('.')[0]
    data_dict = []
    const_sents = io.open(tree_dir, encoding='utf8').read().strip().split('\n\n')
    dep_sents = io.open(dep_dir, encoding='utf8').read().strip().split('\n\n')

    # Due to the conllu format, check if the first block contains meta-data only
    if '# sent_id' not in dep_sents[0] and dataset != 'ontonotes':
        dep_sents = dep_sents[1:]
    if dep_sents[-1] == '#end document':
        dep_sents = dep_sents[:-1]
    if len(const_sents) != len(dep_sents):
        print(f'{filename}: Number of const trees does not match number of dep sentences')
        return None
    for sent_id, t in enumerate(const_sents):
        # test
        # if f'{filename}-{sent_id+1}' != 'GUM_interview_gaming-2':
        #     continue
        # if sent_id < 26:
        #     continue

        # remove function tags for each phrase-level tag
        tree = generate_tree(t, const_type)
        if not tree:
            try:
                tree = generate_tree(f"(S {t})", const_type)
            except:
                raise ValueError(f'Cannot parse sentence "{t}" in {filename}-{sent_id+1}')
        tree.remove_trace()
        if dataset == 'ontonotes':
            tree.remove_edited_tokens()
        if not tree:
            continue
        # tree.remove_leaf_trace()
        try:
            flat_idx_mapping_tree = tree.idx_in_tree()
            np_list = find_np(tree, flat_idx_mapping_tree, [])
        except:
            raise ValueError(f'Error in {filename}-{sent_id + 1}')

        dep_sent = dep_sents[sent_id]
        if dataset == 'ontonotes':
            toks, sent_tok_info, sent_entities = read_ontonotes_conll(dep_sent)
            real_sent_id = sent_id + 1
        else:
            toks, sent_tok_info, sent_entities, real_sent_id = read_conllu(dep_sent)

        if len(sent_tok_info) != len(tree.leaves()):
            ValueError(f'{filename}-{sent_id}: Number of tokens does not match number of tree leaves')
            # continue
        tree.add_info(sent_tok_info, flat_idx_mapping_tree)
        node_labels = get_tree_labels(tree, [])
        for l in node_labels:
            node_label_freq[l] += 1

        labels = create_labels(np_list, sent_entities)

        # reverse check: if mention is NP
        reverse_span_labels = reverse_labels(np_list, sent_entities, toks)
        mention_not_np.append([f'{filename}-{real_sent_id}', reverse_span_labels, tree, toks])

        sent_data = {'id': f'{filename}-{real_sent_id}',
                     'sent': tree.leaves(),
                     'tree': str(tree),
                     'pos': tree.pos,
                     'feats': [],
                     'np_feats': [],
                     'np': [],
                     'np_tag': [],
                     'np_tree': [],
                     'np_span': [],
                     'parent_span': [],
                     'left_siblings_span': [],
                     'right_siblings_span': [],
                     'largest_np_span': [],
                     'smallest_np_span': [],
                     'children_span': [],
                     'nodes': [],
                     'tree_pos': [],
                     'labels': [],
                     }
        for i, np in enumerate(np_list):
            # get span indices of the direct children of the current np span
            children_span = get_children_span(np[2])
            sent_data['children_span'].append(children_span)

            sent_data['np'].append(np_list[i][0])
            sent_data['np_tree'].append(np[2])
            sent_data['np_span'].append(np_list[i][1])
            sent_data['np_tag'].append(np[2].label())
            sent_data['parent_span'].append(np_list[i][2].parent_span)
            sent_data['tree_pos'].append(np_list[i][3])
            sent_data['left_siblings_span'].append(np_list[i][2].left_siblings_span)
            sent_data['right_siblings_span'].append(np_list[i][2].right_siblings_span)
            sent_data['nodes'].append(get_nodes(np[2], []))
            sent_data['labels'].append(labels[i])

            # Find the largest np span over the current np
            sent_data['largest_np_span'].append(find_largest_span(np[2]))

            # Find the smallest np span under the current np
            sent_data['smallest_np_span'].append(find_smallest_span(np[2], np[2].span))

        levels, levels_rels, levels_rels_reverse, largest_levels_rels_reverse = level_of_nps(sent_data['np_span'], sent_data['tree_pos'])

        for i, np in enumerate(np_list):
            # TODO: get id for debugging
            if np[0] == ['course']:
                a = 1
            feats = select_feats(np[2], i, sent_data, levels, levels_rels, levels_rels_reverse, largest_levels_rels_reverse)
            sent_data['feats'].append(feats)

        data_dict.append(sent_data)

    return data_dict


def build_data(const_dir, dep_dir=None, const_type='const'):
    data = []
    dataset = const_dir.split('/')[2]
    for const_filename in tqdm(sorted(os.listdir(const_dir))):
        filename = const_filename.split('.')[0]
        if not filename:
            continue
        # print(filename)
        # if filename != 'GUM_conversation_grounded':
        #     continue
        genre = const_filename.split('_')[0]
        genres[genre] += 1
        pattern = '.conllu'
        if dataset == 'ontonotes':
            pattern = '.v4_gold_conll'
        if dep_dir:
            # print(const_filename)
            file_data = read_file(const_dir+os.sep+const_filename, dep_dir+os.sep+filename+pattern, dataset, const_type)
        else:
            file_data = read_const_file_only(const_dir+os.sep+const_filename, const_type)
        if not file_data:
            print(f'{const_filename} does not pass')
            continue
        data.extend(file_data)
    return data


def build_dataset(data_dir, dataset, corpus, const_type):
    const_dir = data_dir + os.sep + dataset + os.sep + corpus + os.sep + const_type
    dep_dir = data_dir + os.sep + dataset + os.sep + corpus + os.sep + 'dep'
    data = build_data(const_dir, dep_dir, const_type)
    return data


def build_and_save_dataset(data_dir, out_dir, dataset, corpus, const_type):
    const_dir = data_dir + os.sep + dataset + os.sep + corpus + os.sep + const_type
    dep_dir = data_dir + os.sep + dataset + os.sep + corpus + os.sep + 'dep'
    data = build_data(const_dir, dep_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_data(data, os.path.join(out_dir, f'{dataset}_{corpus}.json'))


def build_and_save_unknown_dataset(data_dir, out_dir, dataset, corpus):
    const_dir = data_dir + os.sep + dataset + os.sep + corpus + os.sep + 'const'
    data = build_data(const_dir)
    save_data(data, os.path.join(out_dir, f'{dataset}_{corpus}.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../raw')
    parser.add_argument('--out_dir', default='../data/pred_const/')
    parser.add_argument('--const', default='pred_const', help='const|pred_const')
    parser.add_argument('--datasets', default='ontonotes', help='ontogum|arrau|ontonotes')

    options = parser.parse_args()

    datasets = options.datasets.split('|')
    data_dir = options.data_dir
    out_dir = options.out_dir

    for dataset in datasets:
        # build_and_save_dataset(data_dir, out_dir, dataset, 'train', const_type='const')
        if dataset == 'ontonotes':
            val = 'development'
            build_and_save_dataset(data_dir, out_dir, dataset, val, const_type=options.const)
            # build_and_save_unknown_dataset(data_dir, out_dir, dataset, 'unknown')
        # else:
        #     build_and_save_unknown_dataset(data_dir, out_dir, dataset, 'dev_pred_const')
        build_and_save_dataset(data_dir, out_dir, dataset, 'test', const_type=options.const)
        print(f'Done {dataset}')


if __name__ == '__main__':
    main()
