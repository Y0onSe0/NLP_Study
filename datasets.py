# !/usr/bin/env python3


"""
이 파일은 Quora의 Paraphrase Detection을 위한 Dataset 클래스를 포함한다. 추가 데이터 소스로 훈련시키거나
Quora 데이터셋의 처리 방식(예: 데이터 증강 등)을 변경하려는 경우 이 파일을 수정할 수 있다.
"""

import csv

import re
import torch

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


PARAPHRASE_PROMPT_TEMPLATES = {
  'baseline': 'Question 1: "{q1}"\nQuestion 2: "{q2}"\nAre these questions asking the same thing? Answer "yes" or "no": ',
  'direct': 'Is "{q2}" a paraphrase of "{q1}"? Answer "yes" or "no": ',
  'meaning': 'Do the following two questions have the same meaning?\nQuestion 1: "{q1}"\nQuestion 2: "{q2}"\nAnswer: ',
}


def get_paraphrase_prompt_template(args):
  template_name = getattr(args, 'prompt_template', 'baseline')
  if template_name not in PARAPHRASE_PROMPT_TEMPLATES:
    raise ValueError(f"Unknown prompt_template '{template_name}'. Choose one of: {sorted(PARAPHRASE_PROMPT_TEMPLATES)}")
  return template_name


def build_paraphrase_prompts(sent1, sent2, template_name='baseline'):
  if template_name not in PARAPHRASE_PROMPT_TEMPLATES:
    raise ValueError(f"Unknown prompt_template '{template_name}'. Choose one of: {sorted(PARAPHRASE_PROMPT_TEMPLATES)}")
  template = PARAPHRASE_PROMPT_TEMPLATES[template_name]
  return [template.format(q1=s1, q2=s2) for s1, s2 in zip(sent1, sent2)]


def encode_paraphrase_pairs(tokenizer, sent1, sent2, template_name='baseline', max_length=None):
  cloze_style_sents = build_paraphrase_prompts(sent1, sent2, template_name)
  tokenizer_kwargs = {
    'return_tensors': 'pt',
    'padding': True,
    'truncation': True,
  }
  if max_length is not None and max_length > 0:
    tokenizer_kwargs['max_length'] = max_length
  encoding = tokenizer(cloze_style_sents, **tokenizer_kwargs)
  return torch.LongTensor(encoding['input_ids']), torch.LongTensor(encoding['attention_mask'])


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
    sent_ids = [x[3] for x in all_data]

    template_name = get_paraphrase_prompt_template(self.p)
    token_ids, attention_mask = encode_paraphrase_pairs(
      self.tokenizer, sent1, sent2, template_name, getattr(self.p, 'max_length', None)
    )

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'sent_ids': sent_ids,
      'sent1': sent1,
      'sent2': sent2,
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

    template_name = get_paraphrase_prompt_template(self.p)
    token_ids, attention_mask = encode_paraphrase_pairs(
      self.tokenizer, sent1, sent2, template_name, getattr(self.p, 'max_length', None)
    )

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': sent_ids,
      'sent1': sent1,
      'sent2': sent2,
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
