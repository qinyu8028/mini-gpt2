import csv

import re
import torch

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


def preprocess_string(s):
  return ' '.join(s.lower()
                  .replace('.', ' .')
                  .replace('?', ' ?')
                  .replace(',', ' ,')
                  .replace('\'', ' \'')
                  .split())


class ParaphraseDetectionDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    labels = torch.LongTensor([x[2] for x in all_data])
    # labels = ['yes' if label == 1 else 'no' for label in [x[2] for x in all_data]]
    # labels = self.tokenizer(labels, return_tensors='pt', padding=True, truncation=True)['input_ids']
    sent_ids = [x[3] for x in all_data]

    cloze_style_sents = [f'Question 1: "{s1}"\nQuestion 2: "{s2}\nAre these questions asking the same thing?\n' for
                         (s1, s2) in zip(sent1, sent2)]
    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'sent_ids': sent_ids
    }

    return batched_data


class ParaphraseDetectionTestDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    sent_ids = [x[2] for x in all_data]

    cloze_style_sents = [f'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": ' for (s1, s2) in
                         zip(sent1, sent2)]

    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': sent_ids
    }

    return batched_data


def load_paraphrase_data(paraphrase_filename, split='train'):
  paraphrase_data = []
  if split == 'test':
    with open(paraphrase_filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent_id = record['id'].lower().strip()
        paraphrase_data.append((preprocess_string(record['sentence1']),
                                preprocess_string(record['sentence2']),
                                sent_id))

  else:
    with open(paraphrase_filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        try:
          sent_id = record['id'].lower().strip()
          paraphrase_data.append((preprocess_string(record['sentence1']),
                                  preprocess_string(record['sentence2']),
                                  int(float(record['is_duplicate'])), sent_id))
        except:
          pass

  print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
  return paraphrase_data


class SonnetsDataset(Dataset):
  def __init__(self, file_path):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.sonnets = self._load_sonnets(file_path)

  def _load_sonnets(self, file_path):
    """Reads the file and extracts individual sonnets."""
    with open(file_path, 'r', encoding='utf-8') as f:
      text = f.read()

    # Split sonnets based on numbering pattern (e.g., "\n\n1\n\n")
    sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]  # Remove header text

    # Strip leading/trailing spaces
    return [s.strip() for s in sonnets]

  def __len__(self):
    return len(self.sonnets)

  def __getitem__(self, idx):
    return (idx, self.sonnets[idx])

  def collate_fn(self, all_data):
    idx = [example[0] for example in all_data]
    sonnets = [example[1] for example in all_data]

    encoding = self.tokenizer(sonnets, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': idx
    }

    return batched_data


class HeldOutSonnetsDataset(Dataset):
  def __init__(self, file_path):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.held_out_sonnets = self._load_held_out_sonnets(file_path)

  def _load_held_out_sonnets(self, file_path):
    """Reads the file and extracts individual held out sonnets(first 3 lines)"""
    with open(file_path, 'r', encoding='utf-8') as f:
      text = f.read()

    # Split sonnets based on numbering pattern (e.g., "\n\n1\n\n")
    sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]  # Remove header text
    # Strip leading/trailing spaces
    clean_sonnets = [s.strip() for s in sonnets]
    # extract only the first three lines
    held_out_sonnets = [sonnet.split('\n')[:3] for sonnet in clean_sonnets]

    return held_out_sonnets

  def __len__(self):
    return len(self.held_out_sonnets)

  def __getitem__(self, idx):
    return (idx, self.held_out_sonnets[idx])

  def collate_fn(self, all_data):
    idx = [example[0] for example in all_data]
    sonnets = [example[1] for example in all_data]

    encoding = self.tokenizer(sonnets, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': idx
    }

    return batched_data


class DPOSonnetDataset(Dataset):
  def __init__(self, positive_path, negative_path):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.sonnets_pairs = self._load_sonnets(positive_path, negative_path)

  def _load_sonnets(self, positive_path, negative_path):
    """Reads the file and extracts individual held out sonnets(first 3 lines)"""
    with open(positive_path, 'r', encoding='utf-8') as f:
      positive_text = f.read()
    with open(negative_path, 'r', encoding='utf-8') as f:
      negative_text = f.read()

    # Split sonnets based on numbering pattern (e.g., "\n\n1\n\n")
    positive_sonnets = re.split(r'\n\s*(\d+)\s*\n', positive_text)[1:]  # Remove header text
    negative_sonnets = re.split(r'\n\s*(\d+)\s*\n', negative_text)[1:]
    # negative_sonnets = re.findall(r'\n\n(\d+)([\s\S]+?)\n\n', negative_text)

    # The negative result might have some fake ids in between the lines, e.g., "0730"
    # Thus, must make sure our ids are valid
    negative_numbers = negative_sonnets[0::2]
    negative_contents = negative_sonnets[1::2]

    negative_result = {}
    last_valid_num = 0
    for num_str, content in zip(negative_numbers, negative_contents):
        num = int(num_str)
        if num == last_valid_num + 1:
            negative_result[num] = content
            last_valid_num = num
        else:
            negative_result[last_valid_num] = negative_result[last_valid_num] + '\n' + num_str + '\n' + content
        
    positive_numbers = positive_sonnets[0::2]
    positive_contents = positive_sonnets[1::2]
    positive_result = {int(num_str): content for num_str, content in zip(positive_numbers, positive_contents)}

    sonnets_pairs = []
    for key in positive_result.keys() & negative_result.keys():
      x = positive_result[key].strip().split('\n')[:3]  # extract only the first three lines
      y_win = positive_result[key].strip().split('\n')[3:]
      y_lose = negative_result[key].strip().split('\n')[3:]
      sonnets_pairs.append((x, y_lose, y_win))

    return sonnets_pairs

  def __len__(self):
    return len(self.sonnets_pairs)

  def __getitem__(self, idx):
    return self.sonnets_pairs[idx]

  def collate_fn(self, sonnets_pairs):
    x = ['\n'.join(example[0]) for example in sonnets_pairs]
    negative = ['\n'.join(example[0] + example[1]) for example in sonnets_pairs]
    positive = ['\n'.join(example[0] + example[2]) for example in sonnets_pairs]

    negative_encoding = self.tokenizer(negative, return_tensors='pt', padding=True, truncation=True)
    positive_encoding = self.tokenizer(positive, return_tensors='pt', padding=True, truncation=True)
    x_encoding = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)

    negative_token_ids = torch.LongTensor(negative_encoding['input_ids'])
    negative_attention_mask = torch.LongTensor(negative_encoding['attention_mask'])

    positive_token_ids = torch.LongTensor(positive_encoding['input_ids'])
    positive_attention_mask = torch.LongTensor(positive_encoding['attention_mask'])

    x_len = x_encoding['attention_mask'].sum(dim=1)

    batched_data = {
      'negative_token_ids': negative_token_ids,
      'negative_attention_mask': negative_attention_mask,
      'positive_token_ids': positive_token_ids,
      'positive_attention_mask': positive_attention_mask,
      'x_len': x_len
    }

    return batched_data