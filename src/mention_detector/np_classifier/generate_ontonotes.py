import io
import os
from tqdm import tqdm
from collections import defaultdict
from shutil import copy2
from utils import parse_filenames


ontonotes_dir = '../raw/ontonotes-release-5.0/data/files/data/english/annotations'
conll_dir = '../raw/on_gold_conll/'
out_dir = '../raw/ontonotes'
syntax_files = parse_filenames(ontonotes_dir, pattern='*parse')
conll_files = parse_filenames(conll_dir, pattern='*gold_conll')


genres = defaultdict(int)

# generate coref files
for conll_dir in conll_files:
    f_fields = conll_dir.split('/')
    corpus = f_fields[3]
    genre = '_'.join(f_fields[7:-2])
    filename = f_fields[-1].split('.')[0]
    const_dir = os.path.join(ontonotes_dir, '/'.join(f_fields[7:-1])+os.sep+filename+'.parse')

    dep_path = out_dir + os.sep + corpus + os.sep + 'dep'
    const_path = out_dir + os.sep + corpus + os.sep + 'const'
    if not os.path.exists(dep_path):
        os.makedirs(dep_path)
        os.makedirs(const_path)

    copy2(conll_dir, dep_path + os.sep + f'{genre}_{filename}.v4_gold_conll')
    copy2(const_dir, const_path + os.sep + f'{genre}_{filename}.parse')


# generate the other 1.4M tokens that do not belong to ON-coref
syntax_files = [f for f in syntax_files]
conll_files = set([tuple(f.split('/')[7:-1] + [f.split('/')[-1].split('.')[0]]) for f in conll_files])
const_path = out_dir + os.sep + 'unknown' + os.sep + 'const'
if not os.path.exists(const_path):
    os.makedirs(const_path)

for syntax_dir in syntax_files:
    f_fields = syntax_dir.split('/')
    filename = f_fields[-1].replace('.parse', '')
    genre = '_'.join(f_fields[8:-2])
    file_dir = tuple(f_fields[8:-1] + [filename])
    if file_dir not in conll_files:
        copy2(syntax_dir, const_path + os.sep + f_fields[-1])

print('Done')
