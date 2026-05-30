# !/usr/bin/env python3

"""
Quora paraphrase detection을 위한 평가.

model_eval_paraphrase: 레이블 정보가 있는 dev 및 train dataloader에 적합함.
model_test_paraphrase: 레이블 정보가 없는 test dataloader에 적합.
"""

import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
from sacrebleu.metrics import CHRF
from datasets import (
  encode_paraphrase_pairs,
  get_paraphrase_prompt_template,
  SonnetsDataset,
)

TQDM_DISABLE = False


def _yes_probs_from_logits(logits):
  probs = torch.softmax(logits, dim=1)
  return probs[:, 1]


def _batch_yes_probs(batch, dataloader, model, device, bidirectional=False, prompt_template=None, max_length=None):
  b_ids, b_mask = batch['token_ids'], batch['attention_mask']
  b_ids = b_ids.to(device)
  b_mask = b_mask.to(device)

  logits = model(b_ids, b_mask)
  yes_probs = _yes_probs_from_logits(logits)

  if bidirectional:
    if 'sent1' not in batch or 'sent2' not in batch:
      raise ValueError("Bidirectional paraphrase evaluation requires 'sent1' and 'sent2' in each batch.")
    dataset = dataloader.dataset
    tokenizer = dataset.tokenizer
    if prompt_template is None:
      prompt_template = get_paraphrase_prompt_template(dataset.p)
    if max_length is None:
      max_length = getattr(dataset.p, 'max_length', None)
    rev_ids, rev_mask = encode_paraphrase_pairs(
      tokenizer, batch['sent2'], batch['sent1'], prompt_template, max_length
    )
    rev_ids = rev_ids.to(device)
    rev_mask = rev_mask.to(device)
    rev_logits = model(rev_ids, rev_mask)
    rev_yes_probs = _yes_probs_from_logits(rev_logits)
    yes_probs = (yes_probs + rev_yes_probs) / 2.0

  return yes_probs.detach().cpu().numpy()


def best_threshold_for_accuracy(y_true, yes_probs, threshold_min=0.30, threshold_max=0.70, threshold_step=0.01):
  y_true = np.array(y_true, dtype=int)
  yes_probs = np.array(yes_probs, dtype=float)
  thresholds = np.arange(threshold_min, threshold_max + threshold_step / 2.0, threshold_step)
  best_threshold = 0.5
  best_acc = -1.0
  for threshold in thresholds:
    preds = (yes_probs >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    if acc > best_acc or (acc == best_acc and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
      best_acc = acc
      best_threshold = float(round(threshold, 10))
  return best_threshold, best_acc


@torch.no_grad()
def model_eval_paraphrase(dataloader, model, device, threshold=0.5, bidirectional=False,
                          prompt_template=None, max_length=None):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true, y_pred, sent_ids, y_probs = [], [], [], []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_sent_ids, labels = batch['sent_ids'], batch['labels'].flatten()

    yes_probs = _batch_yes_probs(batch, dataloader, model, device, bidirectional, prompt_template, max_length)
    preds = (yes_probs >= threshold).astype(int).flatten()

    y_true.extend(labels.cpu().numpy().astype(int).tolist())
    y_pred.extend(preds)
    sent_ids.extend(b_sent_ids)
    y_probs.extend(yes_probs.tolist())

  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)

  return acc, f1, y_pred, y_true, sent_ids, y_probs


@torch.no_grad()
def model_test_paraphrase(dataloader, model, device, threshold=0.5, bidirectional=False,
                          prompt_template=None, max_length=None):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_pred, sent_ids, y_probs = [], [], []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_sent_ids = batch['sent_ids']

    yes_probs = _batch_yes_probs(batch, dataloader, model, device, bidirectional, prompt_template, max_length)
    preds = (yes_probs >= threshold).astype(int).flatten()

    y_pred.extend(preds)
    sent_ids.extend(b_sent_ids)
    y_probs.extend(yes_probs.tolist())

  return y_pred, sent_ids, y_probs


def test_sonnet(
    test_path='predictions/generated_sonnets.txt',
    gold_path='data/TRUE_sonnets_held_out.txt'
):
    chrf = CHRF()  # Character n-gram F-score

    # get the sonnets
    generated_sonnets = [x[1] for x in SonnetsDataset(test_path)]
    true_sonnets = [x[1] for x in SonnetsDataset(gold_path)]
    max_len = min(len(true_sonnets), len(generated_sonnets))
    true_sonnets = true_sonnets[:max_len]
    generated_sonnets = generated_sonnets[:max_len]

    # compute chrf
    chrf_score = chrf.corpus_score(generated_sonnets, [true_sonnets])
    return float(chrf_score.score)
