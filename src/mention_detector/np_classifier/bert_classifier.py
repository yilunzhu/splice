import io
import json
import argparse
import os
import time
import random
from tqdm import tqdm
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertModel
from transformers import AdamW
from sklearn import metrics
from sklearn.metrics import classification_report

random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
writer = SummaryWriter()
logger = logging.getLogger()


def set_logger(log_dir):
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logger.setLevel(logging.INFO)

    # set logger directory
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logfile = os.path.join(log_dir, f'bert_classifier_{rq}.log')

    # write to log file
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # write to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def padding(lst, max_len):
    return lst + [0] * (max_len - len(lst))


def batch_sampler(batch):
    batch_input_ids, batch_attention_mask, batch_np_mask, batch_np_span_indices, batch_labels = [], [], [], [], []
    batch_parent_span_mask, batch_left_siblings_span_mask, batch_right_siblings_span_mask = [], [], []
    batch_parent_span_indices, batch_left_siblings_span_indices, batch_right_siblings_span_indices = [], [], []
    batch_children_span_indices, batch_children_span_mask = [], []
    max_batch_len = 0
    for item in batch:
        batch_input_ids.append(item['input_ids'])
        batch_attention_mask.append(item['attention_mask'])
        batch_np_mask.append(item['np_mask'])
        batch_np_span_indices.append(item['np_span_indices'])
        batch_parent_span_mask.append(item['parent_span_mask'])
        batch_parent_span_indices.append(item['parent_span_indices'])
        batch_left_siblings_span_mask.append(item['left_siblings_span_mask'])
        batch_left_siblings_span_indices.append(item['left_siblings_span_indices'])
        batch_right_siblings_span_mask.append(item['right_siblings_span_mask'])
        batch_right_siblings_span_indices.append(item['right_siblings_span_indices'])
        batch_children_span_indices.append(item['children_span_indices'])
        batch_children_span_mask.append(item['children_span_mask'])
        batch_labels.append(item['labels'])
        max_batch_len = len(item['input_ids']) if len(item['input_ids']) > max_batch_len else max_batch_len

    batch_input_ids = torch.LongTensor([padding(lst, max_batch_len) for lst in batch_input_ids])
    batch_attention_mask = torch.LongTensor([padding(lst, max_batch_len) for lst in batch_attention_mask])
    batch_np_mask = torch.LongTensor([padding(lst, max_batch_len) for lst in batch_np_mask])
    batch_np_span_indices = torch.LongTensor(batch_np_span_indices)
    batch_parent_span_mask = torch.LongTensor([padding(lst, max_batch_len) for lst in batch_parent_span_mask])
    batch_parent_span_indices = torch.LongTensor(batch_parent_span_indices)
    batch_left_siblings_span_mask = torch.LongTensor(
        [padding(lst, max_batch_len) for lst in batch_left_siblings_span_mask])
    batch_left_siblings_span_indices = torch.LongTensor(batch_left_siblings_span_indices)
    batch_right_siblings_span_mask = torch.LongTensor(
        [padding(lst, max_batch_len) for lst in batch_right_siblings_span_mask])
    batch_right_siblings_span_indices = torch.LongTensor(batch_right_siblings_span_indices)
    batch_children_span_indices = torch.LongTensor(batch_children_span_indices)
    batch_children_span_mask = torch.LongTensor(
        [[padding(l, max_batch_len) for l in lst] for lst in batch_children_span_mask])
    batch_labels = torch.LongTensor(batch_labels)

    return (
               batch_input_ids,
               batch_attention_mask,
               batch_np_mask,
               batch_np_span_indices,
               batch_parent_span_mask,
               batch_parent_span_indices,
               batch_left_siblings_span_mask,
               batch_left_siblings_span_indices,
               batch_right_siblings_span_mask,
               batch_right_siblings_span_indices,
               batch_children_span_mask,
               batch_children_span_indices
           ), \
           batch_labels


class NPDataset(Dataset):
    """
    The NP Dataset class extracts text spans from build_data.py, below are the features used for model training:
        - current NP span
        - parent NP span
        - left siblings span
        - right siblings span
        - TODO: add other NP spans, e.g. parent (non-direct) and children spans
    """

    def __init__(self, file_dir):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.tokenized_text = []
        self.token_to_subtokens = {}
        self.subtoken_map = []
        self.max_len = 512
        self.d = self._read_file(file_dir)
        self.data = self.dataset_reader()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent_tokens = self.tokenize_with_subtoken_map(self.data[idx][0])
        np_span_indices, \
        parent_span_indices, \
        left_siblings_span_indices, \
        right_siblings_span_indices, \
        children_span_indcies = self.data[idx][2], self.data[idx][3], self.data[idx][4], self.data[idx][5], \
                                self.data[idx][6]
        np_span_mask = self.span_indices_mapping(np_span_indices)
        parent_span_mask = self.span_indices_mapping(parent_span_indices)
        left_siblings_span_mask = self.span_indices_mapping(left_siblings_span_indices)
        right_siblings_span_mask = self.span_indices_mapping(right_siblings_span_indices)
        children_span_mask = self._children_span_indices_mapping(children_span_indcies)

        input_ids = self.tokenizer.convert_tokens_to_ids(sent_tokens)
        attention_mask = [1] * len(input_ids)
        # encoded = self.tokenizer(self.data[idx][1], max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'np_mask': np_span_mask,
                'np_span_indices': np_span_indices,
                'parent_span_mask': parent_span_mask,
                'parent_span_indices': parent_span_indices,
                'left_siblings_span_mask': left_siblings_span_mask,
                'left_siblings_span_indices': left_siblings_span_indices,
                'right_siblings_span_mask': right_siblings_span_mask,
                'right_siblings_span_indices': right_siblings_span_indices,
                'children_span_indices': children_span_indcies,
                'children_span_mask': children_span_mask,
                'labels': self.data[idx][-1]}

    def _children_span_indices_mapping(self, children_span_indcies):
        span_mask = []
        for span in children_span_indcies:
            span_mask.append(self.span_indices_mapping(span))
        return span_mask

    def _get_avg_np_len(self):
        np_len = []
        for line in self.d:
            np_len.append(len(line['np']))
        return sum(np_len) / len(np_len)

    def _read_file(self, file_dir: str):
        with io.open(file_dir, encoding='utf8') as f:
            d = json.load(f)
        return d

    def dataset_reader(self):
        data = []
        # labels = []
        for line in self.d:
            for i in range(len(line['np'])):
                data.append([' '.join(line['sent']),
                             ' '.join(line['np'][i]),
                             line['np_span'][i],
                             line['parent_span'][i],
                             line['left_siblings_span'][i],
                             line['right_siblings_span'][i],
                             line['children_span'][i],
                             line['tree'],
                             line['id'],
                             line['labels'][i],
                             ])
            # labels.append(line['label'])
        return data

    def tokenize_with_subtoken_map(self, text):
        self.subtoken_map = [0]
        self.token_to_subtokens = {}
        count = 1

        self.tokenized_text = ['[CLS]']
        for i, word in enumerate(text.split(' ')):
            tokenized = self.tokenizer.tokenize(word)
            self.tokenized_text += tokenized
            self.subtoken_map += [i + 1] * (len(tokenized))
            self.token_to_subtokens[i] = (count, count + len(tokenized) - 1)
            count += len(tokenized)
        self.tokenized_text += ['[SEP]']
        self.subtoken_map.append(self.subtoken_map[-1] + 1)
        return self.tokenized_text[:self.max_len]

    def span_indices_mapping(self, span):
        # if span is larger than the max length, only use the EDU text
        # span starts from 0 and subtoken_map starts from 1
        # if span[-1][-1]+1 not in subtoken_map[:self.max_length]:
        #     tokenized_span = [Token('[CLS]')]
        #     for i,s in enumerate(span):
        #         tokenized_span += tokenized_text[token_to_subtokens[s[0]][0]:token_to_subtokens[s[1]][-1]+1]
        #         a = 1
        #     tokenized_span += [Token('[SEP]')]
        #     tokenized_text = tokenized_span
        #
        #     span_mask = [1] * len(tokenized_text)
        #     span_mask[0], span_mask[-1] = 0, 0
        #     span = [(1, 511)] if len(tokenized_text) > self.max_length else [(1, len(tokenized_text)-2)]
        #
        # # otherwise use the sentence text
        # else:
        span_mask = [0] * len(self.tokenized_text)
        if span != [-1, -1]:
            s_start, s_end = span
            new_start, new_end = self.token_to_subtokens[s_start][0], self.token_to_subtokens[s_end - 1][-1]
            # span[i] = (new_start, new_end)
            for x in range(new_start, new_end):
                span_mask[x] = 1
        assert len(self.subtoken_map) == len(self.tokenized_text)

        return span_mask[:self.max_len]


def weighted_sum(att, mat):
    if att.dim() == 2 and mat.dim() == 3:
        return att.unsqueeze(1).bmm(mat).squeeze(1)
    elif att.dim() == 3 and mat.dim() == 3:
        return att.bmm(mat)
    else:
        AssertionError('Incompatible attention weights and matrix.')


class NPClassification(nn.Module):
    """
    The classifier encodes the entire sentence via pre-trained language model, and generates span embeddings which
    are extracted by NPDataset.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels

        self._context_layer = BertModel.from_pretrained('bert-base-cased')
        self._global_attention = torch.nn.Linear(config.hidden_size, 1)
        self._weighted_sum = torch.nn.Linear(config.num_children, 1)
        self.dropout = nn.Dropout(config.dropout)

        classifier_dim = 6
        if config.use_siblings:
            self.null_tensor = ((torch.rand(1, config.hidden_size) - 0.5) / 10).to(device)
            classifier_dim += 6
        if config.use_children:
            classifier_dim += 3
        self.classifier = nn.Linear(classifier_dim * config.hidden_size, config.num_labels)
        self.loss_fct = CrossEntropyLoss()

        # self.init_weights()

    def _get_span_embeddings(self, contextualized_unit_sentence, unit_span_mask):
        """
        Get span representations based on attention weights
        """
        global_attention_logits = self._global_attention(contextualized_unit_sentence)  # [b, s, 1]
        concat_tensor = torch.cat([contextualized_unit_sentence, global_attention_logits], -1)  # [b, s, e+1]

        resized_span_mask = torch.unsqueeze(unit_span_mask, -1).expand(-1, -1, concat_tensor.size(-1))
        concat_output = concat_tensor * resized_span_mask
        span_embeddings = concat_output[:, :, :-1]  # [b, s, e]
        span_attention_logits = concat_output[:, :, -1]  # [b, s, 1]

        span_attention_weights = F.softmax(span_attention_logits, 1)  # [b, s]
        attended_text_embeddings = weighted_sum(span_attention_weights, span_embeddings)  # [b, e]
        return attended_text_embeddings

    def _get_end_points_embeddings(self, embedded_sent, span_indices):
        span_indices = span_indices.tolist()

        out_start, out_end = [], []
        for i, span in enumerate(span_indices):
            if span == [-1, -1]:
                out_start.append(self.null_tensor)
                out_end.append(self.null_tensor)
            else:
                start_idx, end_idx = span[0], span[-1]
                cur_start_embedding = torch.unsqueeze(embedded_sent[i][start_idx], 0)
                cur_end_embedding = torch.unsqueeze(embedded_sent[i][end_idx], 0)
                out_start.append(cur_start_embedding)
                out_end.append(cur_end_embedding)
        start_embeddings = torch.cat(out_start, dim=0).to(device)
        end_embeddings = torch.cat(out_end, dim=0).to(device)
        return start_embeddings, end_embeddings

    def _add_null_embeddings(self, span_embeddings):
        indices = (torch.sum(span_embeddings, dim=1) == 0.0).nonzero()
        span_embeddings[indices] = self.null_tensor
        return span_embeddings

    def _get_span_representations(self, embedded_sent, np_span_mask, np_span_indices):
        """
        The span representations are from the end-to-end coreference system (Lee et al., 2017), which includes three
        components: the starting and ending boundary of the span and the soft head representation
        """
        assert np_span_mask.size(1) == embedded_sent.size(1)
        weighted_span_embeds = self._get_span_embeddings(embedded_sent, np_span_mask)  # [b, e]
        weighted_span_embeds = self._add_null_embeddings(weighted_span_embeds)
        start_embeddings, end_embeddings = self._get_end_points_embeddings(embedded_sent, np_span_indices)  # [b, e]
        span_representations = torch.cat([start_embeddings, weighted_span_embeds, end_embeddings], 1)  # [b, 3e]
        return span_representations

    def forward(
            self,
            batch=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        input_ids, attention_mask, np_span_mask, np_span_indices, parent_span_mask, \
        parent_span_indices, left_siblings_span_mask, left_siblings_span_indices, \
        right_siblings_span_mask, right_siblings_span_indices, children_span_mask, \
        children_span_indices = batch
        input_ids = input_ids.squeeze()
        attention_mask = attention_mask.squeeze()

        outputs = self._context_layer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_output = outputs[0]
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        # get np embeddings from hidden embeddings
        np_outputs = self._get_span_representations(hidden_output, np_span_mask, np_span_indices)

        # include parent and left & right sibling embeddings
        parent_outputs = self._get_span_representations(hidden_output, parent_span_mask, parent_span_indices)
        concat_embeds = torch.cat((np_outputs, parent_outputs), dim=-1)

        if self.config.use_siblings:
            left_siblings_outputs = self._get_span_representations(hidden_output, left_siblings_span_mask,
                                                                   left_siblings_span_indices)
            right_siblings_outputs = self._get_span_representations(hidden_output, right_siblings_span_mask,
                                                                    right_siblings_span_indices)
            concat_embeds = torch.cat((concat_embeds, left_siblings_outputs, right_siblings_outputs),
                                      dim=-1)

        if self.config.use_children:
            batch_size, num_children, seq_len = children_span_mask.size(0), children_span_mask.size(
                1), children_span_mask.size(2)
            children_embeddings = torch.cat([self._get_span_representations(hidden_output,
                                                                            children_span_mask[:, idx, :],
                                                                            children_span_indices[:, idx, :])
                                             for idx in range(num_children)])
            # children_embeddings = children_embeddings.to(device)
            children_embeddings = children_embeddings.view(batch_size, num_children, -1)
            children_embeddings = torch.transpose(children_embeddings, 1, 2)
            children_sum_embeddings = self._weighted_sum(children_embeddings).squeeze()
            concat_embeds = torch.cat((concat_embeds, children_sum_embeddings), dim=1)

        concat_embeds = self.dropout(concat_embeds)
        logits = self.classifier(concat_embeds)
        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (
            loss,
            logits,
            outputs.hidden_states,
            outputs.attentions,
        )


def get_metrics_score(y_gold, y_pred):
    p = metrics.precision_score(y_gold, y_pred)
    r = metrics.recall_score(y_gold, y_pred)
    f1 = metrics.f1_score(y_gold, y_pred)
    return p, r, f1


def evaluate(model, dataloader):
    model.eval()
    y_gold, y_pred = [], []
    eval_loss = 0.0
    with torch.no_grad():
        for batch, labels in dataloader:
            batch = [element.to(device) for element in batch]
            labels = labels.to(device)
            outputs = model(batch=batch, labels=labels)
            eval_loss += outputs[0].item()
            _, predictions = torch.max(outputs[1], 1)
            y_pred.extend(predictions.tolist())
            y_gold.extend(labels.tolist())
    return eval_loss, y_gold, y_pred


def predict(model, dataloader):
    # model.eval()
    y_pred = []
    with torch.no_grad():
        for batch, labels in dataloader:
            batch = [element.to(device) for element in batch]
            labels = labels.to(device)
            outputs = model(batch=batch, labels=labels)
            _, predictions = torch.max(outputs[1], 1)
            y_pred.extend(predictions.tolist())
    return y_pred


def train(model, config, train_dataloader, dev_dataloader, model_dir):
    model.train()

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

    total_step = len(train_dataloader)
    min_loss = float('inf')
    best_model = ''
    epoch_not_improve = 0

    for epoch in range(config.epoch):
        epoch_loss, running_loss = 0.0, 0.0
        # pbar = tqdm(enumerate(train_dataloader), desc=f'{epoch+1}/{config.epoch}')

        for idx, (batch, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = [element.to(device) for element in batch]
            labels = labels.to(device)
            step = epoch * total_step + idx

            outputs = model(batch=batch,
                            labels=labels)

            loss = outputs[0]
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # pbar.set_description(f'loss={loss.item()}')
            running_loss += loss.item()

            if (step + 1) % 100 == 0:
                logger.info('Step [{}/{}], Loss: {:.4f}'
                            .format(step + 1, config.epoch * total_step, running_loss))
                # print('Step [{}/{}], Loss: {:.4f}'
                #       .format(step + 1, config.epoch * total_step, running_loss))
                writer.add_scalar('Loss/train', loss.item(), step + 1)
                running_loss = 0.0

        eval_loss, y_gold, y_pred = evaluate(model, dev_dataloader)
        p, r, f1 = get_metrics_score(y_gold, y_pred)
        logger.info(f'---------- Epoch [{epoch + 1}/{config.epoch}]')
        logger.info('Dev set: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}\n'.format(p, r, f1))
        writer.add_scalar('Loss/dev_pred_const', eval_loss, (epoch + 1) * total_step)
        writer.add_scalar('F1/dev_pred_const', f1, (epoch + 1) * total_step)

        ts = time.time()
        model_name = f'model.step_{(epoch + 1) * total_step}.{ts}.pt'
        torch.save(model.state_dict(), model_dir + os.sep + model_name)

        if eval_loss < min_loss:
            best_model = model
            min_loss = eval_loss
        else:
            epoch_not_improve += 1

        # early stopping
        if epoch_not_improve == config.patience:
            break

        torch.save(best_model.state_dict(), model_dir + os.sep + 'best_model.pt')


def main(config):
    set_logger(config.log_dir)
    ts = time.time()

    train_data = NPDataset(config.train_dir)
    dev_data = NPDataset(config.dev_dir)
    test_data = NPDataset(config.test_dir)

    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=batch_sampler)
    dev_dataloader = DataLoader(dev_data, batch_size=config.batch_size, shuffle=True, collate_fn=batch_sampler)
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, collate_fn=batch_sampler)

    model = NPClassification(config)
    model.to(device)

    if config.train:
        model_dir = config.model_dir + os.sep + f'models_{ts}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        train(model, config, train_dataloader, dev_dataloader, model_dir)

    writer.flush()
    writer.close()

    if config.eval_on_test:
        best_model_dir = model_dir + os.sep + 'best_model.pt'
        if config.best_model_dir:
            best_model_dir = config.best_model_dir

        model.eval()
        model.load_state_dict(torch.load(best_model_dir, map_location=device))
        _, y_gold, y_pred = evaluate(model, test_dataloader)
        logger.info(classification_report(y_gold, y_pred, target_names=['0', '1']))

        if config.print_errors:
            f_error = io.open('bert_errors.txt', 'w', encoding='utf8')
            for idx in range(len(y_pred)):
                if y_pred[idx] != y_gold[idx]:
                    f_error.write(f'Gold label: {y_gold[idx]}\n'
                                  f'Pred label: {y_pred[idx]}\n'
                                  f'id: {test_data.data[idx][8]}\n'
                                  f'NP: {test_data.data[idx][1]}\n'
                                  f'Sent: {test_data.data[idx][0]}\n'
                                  f'Tree:\n{test_data.data[idx][7]}\n'
                                  f'---------------\n\n')
            f_error.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True)
    parser.add_argument('--eval_on_test', default=True)
    parser.add_argument('--best_model_dir', default='')
    parser.add_argument('--train_dir', default='../data/arrau_train.json')
    parser.add_argument('--dev_dir', default='../data/arrau_test.json')
    parser.add_argument('--test_dir', default='../data/arrau_test.json')
    parser.add_argument('--model_dir', default='../model')
    parser.add_argument('--hidden_size', default=768)
    parser.add_argument('--num_labels', default=2)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--embed_dim', default=30)
    parser.add_argument('--lr', default=1e-5)
    parser.add_argument('--weight_decay', default=1e-5)
    parser.add_argument('--gamma', default=0.1)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--epoch', default=20)
    parser.add_argument('--patience', default=5)
    parser.add_argument('--num_children', default=5)
    parser.add_argument('--use_siblings', default=True)
    parser.add_argument('--use_children', default=True)
    parser.add_argument('--print_errors', default=True)
    parser.add_argument('--log_dir', default='../log')

    config = parser.parse_args()

    main(config)
