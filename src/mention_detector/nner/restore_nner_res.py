import io
import os
import re
import json
import pickle
from tqdm import tqdm

data_type = 'on-sg'
corpus = ['dev', 'test']
out_dir = f'./out/ontonotes/{data_type}'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for c in corpus:
    gold_path = f'./data/ontonotes/{c}.english.v4_gold_conll'
    gold_sents = io.open(gold_path, encoding='utf8').read().split('\n\n')

    pred_path = f'./preds/ontonotes/on-sg/predictions_{c}_epoch_0.json'

    with io.open(pred_path, encoding='utf8') as f:
        pred_data = json.load(f)

    out_data = []
    sent_id = 0

    total_gold_ent = 0
    total_pred_ent = 0
    found_ent = 0
    docname = ''
    ent_id = 1
    new_doc = []

    pred_sent_id = 0
    for gold_sent_id, sent in tqdm(enumerate(gold_sents[:-1])):
        if sent.startswith('#begin document'):
            docname = re.search('#begin document \(([\w\W]+?)\);', sent).group(1)
            ent_id = 1
        gold_sent = sent.split('\n')
        
        if pred_data[pred_sent_id]['tokens'] == ['(', 'end', ')']:
            pred_sent_id += 1
        pred_sent = pred_data[pred_sent_id]
        pred_toks = pred_sent['tokens']

        sent_pos = []
        heading = []
        lines = []
        for line in gold_sent:
            if line.startswith('#'):
                heading.append(line)
            else:
                line_fields = line.split()
                lines.append(line_fields)
                sent_pos.append(line_fields[4])

        gold_toks = [l[3] for l in lines]
        if pred_toks != gold_toks:
            a = 1
        if pred_sent_id == len(pred_data) - 3:
            a = 1
        
        pred_ents = []
        for e in pred_sent['entities']:
            ent = [e['start'], e['end'] - 1]
            if ent not in pred_ents:
                pred_ents.append(ent)

        for idx in range(1, len(lines) - 1):
            prev_pos = sent_pos[idx - 1]
            pos = sent_pos[idx]
            next_pos = sent_pos[idx + 1]

        coref_fields = [''] * len(lines)
        for start, end in pred_ents:
            if start == end:
                coref_fields[start] += f'|({ent_id})'
            else:
                coref_fields[start] += f'|({ent_id}'
                coref_fields[end] += f'|{ent_id})'
            ent_id += 1

        for i in range(len(lines)):
            lines[i][-1] = coref_fields[i].strip('|') if coref_fields[i] else '-'

        new_sent = '\n'.join(heading) + '\n' + '\n'.join(['     '.join(l) for l in lines])
        new_doc.append(new_sent.strip())
        pred_sent_id += 1
    
    new_doc.append('#end document\n')
    new_doc = '\n\n'.join(new_doc)
    with io.open(os.path.join(out_dir, f'{c}.english.v4_gold_conll'), 'w', encoding='utf8') as f:
        f.write(new_doc)

print('Done!')
