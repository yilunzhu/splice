import io
import os
import shutil

train_out_dir = '../raw/gum/train'
dev_out_dir = '../raw/gum/dev_pred_const'
test_out_dir = '../raw/gum/test'

const_dir = '/Users/yilun/Documents/GeorgetownUniversity/Corpling_lab/gum/_build/target/const'
dep_dir = '/Users/yilun/Documents/GeorgetownUniversity/Corpling_lab/gum/_build/target/coref/ontogum/conllu'
split_dir = 'splits'

docs = {}
for f in os.listdir(split_dir):
    if 'train' in f:
        ftype = 'train'
    elif 'dev_pred_const' in f:
        ftype = 'dev_pred_const'
    else:
        ftype = 'test'
    lines = io.open(os.path.join(split_dir, f), encoding='utf8').read().strip().split('\n')
    for line in lines:
        docs[line] = ftype

def get_split(f_dir, docs):
    if 'const' in f_dir:
        d_type = 'const'
    else:
        d_type = 'dep'
    for filename in os.listdir(f_dir):
        if '.conllu' not in filename:
            continue
        fname = filename.split('.')[0]
        out_dir = f'../raw/gum/{docs[fname]}/{d_type}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        shutil.copy(os.path.join(f_dir, filename), os.path.join(out_dir, filename))

get_split(const_dir, docs)
get_split(dep_dir, docs)
print('Done')
