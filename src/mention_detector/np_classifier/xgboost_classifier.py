import numpy as np
from xgboost import XGBClassifier
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from build_conll import merge_preds
from build_data import *


def padding(lst, max_len):
    return lst + ['PAD'] * (max_len - len(lst))


def data_reader(raw_data, is_training: bool, seed: int):
    all_data, data, all_labels = [], [], []
    for line in raw_data:
        # all_data.append(line)
        data.extend(line['feats'])
        all_labels.extend(line['labels'])

    if is_training:
        data, y = shuffle(data, all_labels, random_state=seed)
    else:
        data, y = data, all_labels

    return np.array(data), np.array(y)


def post_process(sent, idx):
    np_span = sent['np_span'][idx]
    if 'np_tag' not in sent:
        a = 1
    np_tag = sent['np_tag'][idx]
    tokens = sent['sent'][np_span[0]: np_span[1]]
    pos = sent['pos'][np_span[0]:np_span[1]]

    # regard all NML that only contains NNP as mention candidates
    if np_tag == 'NML':
        for t in pos:
            if 'NNP' not in t:
                return 0
        return 1
    # regard all definite NP as mention cadidates
    if len(tokens) == 1 and 'PRP' in pos[0]:
        return 1
    # if tokens[0].lower() in definite_markers:
    #     return 1
    # elif len(tokens) == 1 and tokens[0][0].isupper():
    #     return 1
    return 0


def get_errors(y_pred, y_gold, data):
    errors = []
    zero = []
    np_idx = 0
    for line in data:
        all_correct = True
        for i in range(len(line['np'])):
            fields = {'gold': y_gold[np_idx], 'pred': y_pred[np_idx], 'tree': line['tree'], 'sent': line['sent'],
                      'np': line['np'][i], 'id': line['id'], 'largest_np_span': line['largest_np_span'][i],
                      'smallest_np_span': line['smallest_np_span'][i]}
            if y_pred[np_idx] != y_gold[np_idx]:
                errors.append(fields)
                all_correct = False
                # if y_pred[np_idx] == 0:
                if len(line['np'][i]) == line['largest_np_span'][i][1] - line['largest_np_span'][i][0]:
                    a = 1
                elif len(line['np'][i]) != line['largest_np_span'][i][1] - line['largest_np_span'][i][0] and \
                        len(line['np'][i]) != line['smallest_np_span'][i][1] - line['smallest_np_span'][i][0]:
                    a = 1
                else:
                    a = 1
            else:
                if y_pred[np_idx] == 0 and y_gold[np_idx] == 0:
                    zero.append(fields)
                    a = 1
            np_idx += 1
        if all_correct and len(line['sent']) < 15 and 0 in y_gold[np_idx-len(line['np'])+1:np_idx+1]:
            a = 1
    return errors, y_pred, zero


def evaluate(model, X, y, data):
    predictions = predict(model, X, data)
    errors, predictions, zero = get_errors(predictions, y, data)
    print(classification_report(y, predictions, target_names=['0', '1']))
    return errors, predictions, zero


def predict(model, X, data):
    y_pred = model.predict(X)
    predictions = [round(value) for value in y_pred]
    return predictions


def write2file(lines, filename='xgboost_errors'):
    with io.open(f'{filename}.txt', 'w', encoding='utf8') as f:
        for line in lines:
            f.write(f'Gold label: {line["gold"]}\nPred label: {line["pred"]}\nid:{line["id"]}\n')
            f.write(' '.join(line['np']))
            f.write('\n')
            f.write(f'largest_np_span: {line["largest_np_span"]}\n')
            f.write(f'smallest_np_span: {line["smallest_np_span"]}\n')
            f.write(' '.join(line['sent']))
            f.write('\n')
            f.write(line['tree'])
            f.write('\n\n')


def train(data, config):
    X_train, y_train = data_reader(data, is_training=True, seed=config.seed)

    model = XGBClassifier(learning_rate=0.1, booster='gbtree', colsample_bytree=0.8, seed=42)
    model.fit(X_train, y_train)
    return model


def compare(og_errors, arrau_errors):
    og_files = [l['id'] for l in og_errors]
    arrau_files = [l['id'] for l in arrau_errors]

    error_not_in_arrau = []
    for idx, id in enumerate(og_files):
        if id not in arrau_files:
            error_not_in_arrau.append(og_errors[idx])

    error_not_in_og = []
    for idx, id in enumerate(arrau_files):
        if id not in og_files:
            error_not_in_og.append(arrau_errors[idx])

    return error_not_in_og, error_not_in_arrau


def output2conll(preds, data):
    np_idx = 0
    mention_idx = 1
    docs = {}
    seen_doc = []
    for idx, line in enumerate(data):
        docname = line['id'].split('-')[0]
        sent_id = line['id'].split('-')[1]

        if docname == 'mz_sinorama_ectb_1029':
            a = 1
        if f'{docname}-{sent_id}' == 'mz_sinorama_ectb_1029-35':
            a = 1

        if docname not in seen_doc:
            seen_doc.append(docname)
            docs[docname] = []
            mention_idx = 1
        sent_np = []
        corefs = []
        for i in range(len(line['np'])):
            if preds[np_idx]:
                if line['np_span'][i] not in sent_np:
                    sent_np.append(line['np_span'][i])
            else:
                a = 1
            if line['labels'] and line['labels'][i]:
                if line['np_span'][i] not in corefs:
                    corefs.append(line['np_span'][i])
            np_idx += 1

        # build conll format
        sent_len = len(line['sent'])
        sent_mention = [''] * sent_len
        for i in sent_np:
            start, end = i
            # if the mention is one token long
            if end - start == 1:
                sent_mention[start] += f'|({mention_idx})'
            else:
                sent_mention[start] += f'|({mention_idx}'
                sent_mention[end-1] += f'|{mention_idx})'
            mention_idx += 1
        sent_mention = [i.strip('|') for i in sent_mention]
        pred_sent = {'id': line['id'],
                     'mention': sent_mention,
                     'span': sent_np,
                     'sent': line['sent'],
                     'tree': line['tree'],
                     'np_tag': line['np_tag'],
                     'pos': line['pos'],
                     'corefs': corefs
                     }
        docs[docname].append(pred_sent)

    return docs


def predict_and_build_conll(model, on_data, corpus, out_dir, const, config, dataset):
    X_on, _ = data_reader(on_data, is_training=False, seed=config.seed)
    predictions = predict(model, X_on, on_data)
    json_data = output2conll(predictions, on_data)

    if dataset == 'ontonotes':
        data_dir = config.gold_conll_dir+os.sep+corpus+os.sep+'data/english/annotations'
    elif dataset == 'ontogum':
        data_dir = config.gold_conll_dir+os.sep+corpus+os.sep+'dep'
    else:
        raise ValueError(f'Do not support {dataset}, available datasets: ontonotes, ontogum')

    # build conll
    merge_preds(json_data,
                data_dir,
                out_dir,
                corpus=corpus,
                const=const,
                dataset=dataset
                )


def main(config):
    const_type = 'const' if config.const == 'gold' else 'pred_const'

    og_dev_data = build_dataset(config.data_dir, dataset='ontogum', corpus='dev', const_type=const_type)
    og_test_data = build_dataset(config.data_dir, dataset='ontogum', corpus='test', const_type=const_type)

    train_data = build_dataset(config.data_dir, dataset='arrau', corpus='train', const_type='const')
    test_data = build_dataset(config.data_dir, dataset='arrau', corpus='test', const_type='const')
    # on_train_data = build_dataset(config.data_dir, dataset='ontonotes', corpus='train', const_type='const')
    # on_dev_data = build_dataset(config.data_dir, dataset='ontonotes', corpus='development', const_type=const_type)
    on_test_data = build_dataset(config.data_dir, dataset='ontonotes', corpus='test', const_type=const_type)
    # on_unkonwn_data = build_dataset(config.data_dir, dataset='ontonotes', corpus='unknown', const_type='const')


    X_test, y_test = data_reader(test_data, is_training=False, seed=config.seed)
    X_on_test, y_on_test = data_reader(on_test_data, is_training=False, seed=config.seed)

    model = train(train_data, config=config)
    errors, preds, zero = evaluate(model, X_on_test, y_on_test, on_test_data)
    write2file(errors, 'errors')
    write2file(zero, "zero")

    print('Predicting...')

    if config.const == 'gold':
        const = 'gold_const'
    elif config.const == 'pred':
        const = 'pred_const'

    # predict_and_build_conll(model, on_train_data, corpus='train', out_dir=config.out, const=const, config=config, dataset='ontonotes')
    # predict_and_build_conll(model, on_dev_data, corpus='development', out_dir=config.out, const=const, config=config, dataset='ontonotes')
    predict_and_build_conll(model, on_test_data, corpus='test', out_dir=config.out, const=const, config=config, dataset='ontonotes')
    # predict_and_build_conll(model, on_unkonwn_data, corpus='unknown', out_dir=config.out, const=const, config=config, dataset='ontonotes')

    # predict_and_build_conll(model, og_dev_data, corpus='dev', out_dir=config.out, const=const, config=config, dataset='ontogum')
    # predict_and_build_conll(model, og_test_data, corpus='test', out_dir=config.out, const=const, config=config, dataset='ontogum')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../raw')
    parser.add_argument('--const', default='pred', help='const type, from [gold|pred]')
    parser.add_argument('--gold_conll_dir', default='../raw/on_gold_conll')
    parser.add_argument('--out', default='../data/ontonotes_sg_pred_const')
    parser.add_argument('--seed', default=42)

    config = parser.parse_args()
    main(config)
    print('Done!')
