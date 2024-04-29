import io
import os
import json
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input_dir', default='pred')
parser.add_argument('--out_dir', default='pred', help='Path to the output directory')
args = parser.parse_args()

input_dir = args.input_dir
out_dir = args.out_dir
f_name = input_dir.split('/')[-1]

with io.open(input_dir, encoding='utf-8') as f:
    lines = f.read().strip().split('\n')
    docs = []

    for example_num, line in tqdm(enumerate(lines)):
        example = json.loads(line)
        doc = '/'.join(example['doc_key'].split('/'))
        docname, part = '_'.join(example['doc_key'].split('_')[:-1]), example['doc_key'].split('_')[-1]
        if len(part) == 1:
            part = '00' + part
        elif len(part) == 2:
            part = '0' + part

        # if docname != 'nw/wsj/23/wsj_2351':
        #     continue

        count = 0
        merged = 0
        tsv = []
        tok_map = {}
        tok_count = 1
        prev_sent = 1
        cur_sent_subtok = 0

        # get token and sentence information
        toks = [y for x in example['sentences'] for y in x]
        for i in range(len(example['sentence_map'])):
            cur_subtok = example['subtoken_map'][i]
            cur_sent = example['sentence_map'][i] + 1
            if cur_sent != prev_sent:
                tok_count = 1
                prev_sent = cur_sent
                cur_sent_subtok = cur_subtok
            tok_map[i] = {'sent': cur_sent,
                          'tok': toks[i],
                          'sub_tok': example['subtoken_map'][i],
                          'tok_id': tok_count,
                          'sent_subtok': example['subtoken_map'][i] - cur_sent_subtok + 1
                          }
            tok_count += 1

        # generate mention spans and which span corefers to which one
        # coref {1:2, 2:3}
        # mentions {331:1, 332:1, 339:2}
        # mention_start {1:True}
        # mention_start_pos {1:331}
        predicted_clusters = example['predicted_clusters'] if 'predicted_clusters' in example else example['clusters']
        coref, mentions, mention_start, mention_start_pos = {}, {}, {}, {}

        coref_lines = defaultdict(str)
        mention_num = 1
        for idx, cluster in enumerate(predicted_clusters):
            c_id = idx + 1
            for m in cluster:
                subtok_start, subtok_end = m
                tok_start, tok_end = example['subtoken_map'][subtok_start], example['subtoken_map'][subtok_end]
                if tok_start == tok_end:
                    coref_lines[tok_start] += f"|({c_id})"
                else:
                    coref_lines[tok_start] += f"|({c_id}"
                    coref_lines[tok_end] += f"|{c_id})"
        a = 1

        # generate conll format
        prev_sent_id = 1
        count = 0
        sent, tok_id = [], 0
        last_subtok = -1
        sents = []
        tok_span = []
        for i in tok_map.keys():
            sent_id = tok_map[i]['sent']
            cur_tok = tok_map[i]['tok']
            cur_subtok = tok_map[i]['sub_tok']

            if cur_tok in ['[CLS]', '[SEP]']:
                continue
            if cur_subtok == last_subtok:
                sent[-1][3] += cur_tok.replace('#', '')
                continue

            tok_id += 1

            if sent_id != prev_sent_id:
                sent.append('')
                sents.append(["\t".join(l) for l in sent])
                sent = []
                tok_id = 1

            line = [f'{docname}', str(part), str(tok_id), cur_tok, '-', '-', '-', '-', '-', '-', f"*", '*', '*',
                    '*', '*', '*', coref_lines[count].strip('|') if count in coref_lines else '-']

            sent.append(line)
            prev_sent_id = sent_id
            last_subtok = cur_subtok
            count += 1


        sents.append(["\t".join(l) for l in sent])
        sents = f"#begin document ({docname}); part {part}\n" + "\n".join(["\n".join(sent) for sent in sents]) + "\n\n#end document"
        docs.append(sents)
    docs = "\n".join(docs)

with open(out_dir + os.sep + ".".join(f_name.split(".")[:-1]) + ".v4_gold_conll", "w", encoding="utf-8") as f:
    f.write(docs)
