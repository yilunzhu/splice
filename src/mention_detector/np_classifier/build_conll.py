import io
import os
from collections import defaultdict
from utils import parse_filenames, add_entities

IS_ORACLE = False
POST_PROCESSING = True
if POST_PROCESSING:
    EXTRACT_NNP = True
    EXTRACT_VERB = False
    EXTRACT_DT_NP = False
    EXTRACT_NUM = True
else:
    EXTRACT_NNP = False
    EXTRACT_VERB = False
    EXTRACT_DT_NP = False
    EXTRACT_NUM = False
PRINT_ERRORS = True


def merge_preds(pred_files, data_dir, out_dir, corpus, const, dataset):
    total_gold_ent = 0
    total_pred_ent = 0
    found_ent = 0
    total_sents = 0

    errors = []

    if dataset == 'ontonotes':
        conll_files = parse_filenames(data_dir, pattern='*gold_conll')
    else:
        conll_files = parse_filenames(data_dir, pattern='*conllu')

    for fname in conll_files:
        f_fields = fname.split('/')
        corpus = f_fields[3]
        filename = f_fields[-1].split('.')[0]
        if dataset == 'ontonotes':
            genre = '_'.join(f_fields[7:-2])
            pred_fname = f'{genre}_{filename}'
        else:
            pred_fname = filename

        if pred_fname == 'mz_sinorama_ectb_1070':
            a = 1

        pred_file = pred_files[pred_fname]

        out_file = []

        gold_sents = io.open(fname, encoding='utf8').read().strip().split('\n\n')

        sent_idx = 0
        mention_idx = 0
        for sent_id, sent in enumerate(gold_sents):
            if '#begin document' in sent:
                mention_idx = 0
            if sent == '#end document':
                out_file.append(sent)
                continue
            pred_sent = pred_file[sent_id]
            lines = sent.split('\n')
            tok_id = 0
            toks = []
            sent_entities = {}
            count_same_entities = defaultdict(int)

            stack = []
            for line in lines:
                if line.startswith('#'):
                    continue
                fields = line.split()
                if '-' in fields[0] or '.' in fields[0]:
                    continue
                if 'gum' in dataset:
                    es = [f for f in fields[-1].split('|') if 'Entity=' in f]
                    if es:
                        es = es[0].strip('Entity=').split('|')[0].replace(')(', ')|(').replace(')', ')|').replace('(','|(').replace('||', '|').replace('||', '|').strip('|')
                        fields[-1] = es
                    else:
                        fields[-1] = ''
                sent_entities, count_same_entities = add_entities(fields, sent_entities, tok_id, count_same_entities)
                if dataset == 'ontonotes':
                    token = fields[3]
                else:
                    token = fields[1]
                toks.append(token)
                tok_id += 1

            if tok_id != len(pred_sent['mention']):
                a = 1
            assert tok_id == len(pred_sent['mention']), f'File: {pred_fname}\tSent id: {sent_id}\nGold sent:\n{toks}\n\nPred sent:\n{pred_sent["sent"]}'

            gold_ent = [fields['boundary'] for fields in sent_entities.values()]
            old_pred_ent = pred_sent['span']

            # remove duplicates
            pred_ent = []
            for e in old_pred_ent:
                if e not in pred_ent and len(e) == 2:
                    pred_ent.append(e)

            """
            for idx in range(len(pred_sent['pos'])):
                prev_prev_pos = '' if idx <= 1 else pred_sent['pos'][idx-2]
                prev_pos = '' if idx == 0 else pred_sent['pos'][idx-1]
                next_pos = '' if idx == len(pred_sent['pos'])-1 else pred_sent['pos'][idx+1]
                next_tok = '' if idx == len(pred_sent['pos'])-1 else pred_sent['sent'][idx+1]
                pos = pred_sent['pos'][idx]
                tok = pred_sent['sent'][idx]

                # if EXTRACT_VERB:
                #     if 'VB' in pos:
                #         span = [idx, idx+1]
                #         if span not in pred_ent:
                #             pred_ent.append(span)
                # # extract NNP and verbs
                # if EXTRACT_NNP:
                #     if 'NNP' not in prev_pos and 'NNP' in pos and 'NNP' not in next_pos:
                #         span = [idx, idx+1]
                #         if span not in pred_ent:
                #             pred_ent.append(span)
                #     # if 'N' not in prev_prev_pos and 'NNP' in prev_pos and 'N' in pos and 'N' not in next_pos:
                #     #     span = [idx-1, idx+1]
                #     #     if span not in pred_ent:
                #     #         pred_ent.append(span)
                # if EXTRACT_DT_NP:
                #     if 'DT' in prev_pos and 'N' in pos and 'N' not in next_pos:
                #         span = [idx-1, idx+1]
                #         if span not in pred_ent:
                #             pred_ent.append(span)
                #
                # # extract numbers like "70 %" and "1989"
                # if EXTRACT_NUM:
                #     if 'CD' in prev_pos and 'NN' in pos and tok == '%':
                #         span = [idx-1, idx+1]
                #         if span not in pred_ent:
                #             pred_ent.append(span)
                #     if 'CD' in pos and len(tok) == 4 and tok.isdigit() and next_tok != '%':
                #         span = [idx, idx+1]
                #         if span not in pred_ent:
                #             pred_ent.append(span)
            """

            total_gold_ent += len(gold_ent)
            total_pred_ent += len(pred_ent)

            # tok_lines = [l for l in lines if not l.startswith('#')]

            for e in gold_ent:
                if len(e) != 2:
                    continue
                if tuple(e) in pred_ent:
                    found_ent += 1
                else:
                    if IS_ORACLE:   # oracle
                        pred_ent.append(e)
                        total_pred_ent += 1
                        found_ent += 1
                    else:
                        a = 1
                        try:
                            errors.append((e, toks[e[0]:e[1]], sent, pred_sent['tree']))
                        except:
                            a = 1
            sent_idx += 1

            sent_len = len(pred_sent['sent'])
            sent_mention = [''] * sent_len
            for i in pred_ent:
                start, end = i
                # if the mention is one token long
                if end - start == 1:
                    sent_mention[start] += f'|({mention_idx})'
                else:
                    sent_mention[start] += f'|({mention_idx}'
                    sent_mention[end - 1] += f'|{mention_idx})'
                mention_idx += 1
            sent_mention = [i.strip('|') for i in sent_mention]

            tok_id = 0
            for line in lines:
                if line.startswith('#'):
                    out_file.append(line)
                    continue
                fields = line.split()
                if '-' in fields[0] or '.' in fields[0]:
                    continue
                fields[-1] = '-'
                try:
                    sent_mention[tok_id]
                except:
                    a = 1
                if sent_mention[tok_id]:
                    fields[-1] = sent_mention[tok_id]
                out_file.append('    '.join(fields))
                tok_id += 1
            out_file.append('')

        total_sents += sent_idx
        # save file
        out_string = '\n'.join(out_file) + '\n'
        out_f_dir = out_dir + os.sep + '/'.join(f_fields[3:-1])
        if corpus in ['development', 'test']:
            out_f_dir = out_f_dir.replace(corpus, f'{corpus}_{const}')
        if not os.path.exists(out_f_dir):
            os.makedirs(out_f_dir)
        with io.open(out_f_dir+os.sep+f_fields[-1], 'w', encoding='utf8') as f:
            f.write(out_string)

    # stats
    precision = found_ent / total_pred_ent
    recall = found_ent / total_gold_ent
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f'---------------- {corpus} ----------------')
    print(f'Precision: {precision:.4f} = {found_ent} / {total_pred_ent}')
    print(f'Recall: {recall:.4f} = {found_ent} / {total_gold_ent}')
    print(f'F1: {f1:.4f}')

    if PRINT_ERRORS and corpus != 'train':
        with io.open(f'../errors/{dataset}_{corpus}_errors.txt', 'w', encoding='utf8') as f:
            for e in errors:
                f.write(f'{e[0]}\n')
                f.write(f'{e[1]}\n')
                f.write(e[2] + '\n')
                f.write(e[3] + '\n\n')


if __name__ == '__main__':
    merge_preds('../preds/onto_test_preds.json',
                '../raw/on_gold_conll/test/data/english/annotations',
                '../data/ontonotes_sg',
                corpus='test',
                const='gold',
                dataset='ontonotes'
                )
    merge_preds('../preds/onto_dev_preds.json',
                '../raw/on_gold_conll/development/data/english/annotations',
                '../data/ontonotes_sg',
                corpus='dev_pred_const',
                const='gold',
                dataset='ontonotes'
                )
    merge_preds('../preds/onto_train_preds.json',
                '../raw/on_gold_conll/train/data/english/annotations',
                '../data/ontonotes_sg',
                corpus='train',
                const='gold',
                dataset='ontonotes'
                )
