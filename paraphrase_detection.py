'''
Paraphrase detection을 위한 시작 코드.

고려 사항:
 - ParaphraseGPT: 여러분이 구현한 GPT-2 분류 모델 .
 - train: Quora paraphrase detection 데이터셋에서 ParaphraseGPT를 훈련시키는 절차.
 - test: Test 절차. 프로젝트 결과 제출에 필요한 파일들을 생성함.

실행:
  `python paraphrase_detection.py --use_gpu`
ParaphraseGPT model을 훈련 및 평가하고, 필요한 제출용 파일을 작성한다.
'''

import argparse
import csv
import os
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer

from datasets import (
  PARAPHRASE_PROMPT_TEMPLATES,
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import best_threshold_for_accuracy, model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class ParaphraseGPT(nn.Module):
  """Paraphrase Detection을 위해 설계된 여러분의 GPT-2 Model."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.no_token_id = args.no_token_id
    self.yes_token_id = args.yes_token_id

    # 기본적으로, 전체 모델을 finetuning 한다.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    GPT-2의 마지막 non-padding 토큰 hidden state로부터 vocab logit을 계산하고,
    "yes"와 "no" 토큰의 logit만 추출하여 [batch_size, 2] 형태로 반환합니다.
    """
    gpt_outputs = self.gpt(input_ids, attention_mask)
    last_token_hidden = gpt_outputs['last_token']  # [batch_size, hidden_size]

    # hidden state -> vocab logits
    vocab_logits = self.gpt.hidden_state_to_token(last_token_hidden)  # [batch_size, vocab_size]

    no_logits = vocab_logits[:, self.no_token_id]
    yes_logits = vocab_logits[:, self.yes_token_id]

    # [batch_size, 2] 크기로 결합
    logits = torch.stack([no_logits, yes_logits], dim=1)
    return logits



def verify_yes_no_tokens(args):
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  yes_ids = tokenizer.encode('yes')
  no_ids = tokenizer.encode('no')
  spaced_yes_ids = tokenizer.encode(' yes')
  spaced_no_ids = tokenizer.encode(' no')

  if len(yes_ids) != 1 or len(no_ids) != 1:
    raise ValueError(f'Expected single-token yes/no encodings, got yes={yes_ids}, no={no_ids}')

  args.yes_token_id = yes_ids[0]
  args.no_token_id = no_ids[0]
  print(f'yes/no token ids: yes={args.yes_token_id}, no={args.no_token_id}; '
        f'spaced variants: yes={spaced_yes_ids}, no={spaced_no_ids}')
  return args


def maybe_limit_data(data, max_examples):
  if max_examples is not None and max_examples > 0:
    return data[:max_examples]
  return data


def ensure_output_dirs(args):
  if args.filepath:
    os.makedirs(os.path.dirname(args.filepath) or '.', exist_ok=True)
  for output_path in [args.para_dev_out, args.para_test_out]:
    if output_path:
      os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)


def save_model(model, optimizer, args, filepath, extra_info=None):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }
  if extra_info is not None:
    save_info['extra_info'] = extra_info

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def load_checkpoint(filepath):
  try:
    return torch.load(filepath, weights_only=False)
  except TypeError:
    return torch.load(filepath)


def write_paraphrase_predictions(output_path, sent_ids, predictions):
  with open(output_path, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for sent_id, pred in zip(sent_ids, predictions):
      f.write(f"{sent_id}, {int(pred)} \n")


def record_experiment(args, mode, metrics):
  if not args.experiment_log:
    return
  os.makedirs(os.path.dirname(args.experiment_log) or '.', exist_ok=True)
  fieldnames = [
    'output_tag', 'mode', 'checkpoint', 'prompt_template', 'bidirectional', 'threshold',
    'lr', 'batch_size', 'grad_accum_steps', 'epochs', 'max_train_examples', 'max_dev_examples',
    'max_length', 'dev_acc', 'dev_f1', 'selected_threshold', 'para_dev_out', 'para_test_out',
  ]
  row = {
    'output_tag': args.output_tag,
    'mode': mode,
    'checkpoint': args.filepath,
    'prompt_template': args.prompt_template,
    'bidirectional': args.bidirectional,
    'threshold': args.threshold,
    'lr': args.lr,
    'batch_size': args.batch_size,
    'grad_accum_steps': args.grad_accum_steps,
    'epochs': args.epochs,
    'max_train_examples': args.max_train_examples,
    'max_dev_examples': args.max_dev_examples,
    'max_length': args.max_length,
    'dev_acc': metrics.get('dev_acc'),
    'dev_f1': metrics.get('dev_f1'),
    'selected_threshold': metrics.get('selected_threshold'),
    'para_dev_out': args.para_dev_out,
    'para_test_out': args.para_test_out,
  }
  write_header = not os.path.exists(args.experiment_log)
  with open(args.experiment_log, 'a', newline='') as fp:
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    if write_header:
      writer.writeheader()
    writer.writerow(row)


def build_paraphrase_dataloader(dataset, batch_size, shuffle):
  return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=dataset.collate_fn)


def train(args):
  """Quora 데이터셋에서 Paraphrase Detection을 위한 GPT-2 훈련."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # 데이터, 해당 데이터셋 및 데이터로드 생성하기.
  para_train_data = maybe_limit_data(load_paraphrase_data(args.para_train), args.max_train_examples)
  para_dev_data = maybe_limit_data(load_paraphrase_data(args.para_dev), args.max_dev_examples)

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = build_paraphrase_dataloader(para_train_data, args.batch_size, shuffle=True)
  para_dev_dataloader = build_paraphrase_dataloader(para_dev_data, args.batch_size, shuffle=False)

  model = ParaphraseGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)
  best_dev_acc = -1.0

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
      # 입력을 가져와서 GPU로 보내기(이 모델을 CPU에서 훈련시키는 것을 권장하지 않는다).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      # 손실, 그래디언트를 계산하고 모델 파라미터 업데이트. 
      logits = model(b_ids, b_mask)
      loss = F.cross_entropy(logits, labels, reduction='mean')
      (loss / args.grad_accum_steps).backward()

      if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(para_train_dataloader):
        optimizer.step()
        optimizer.zero_grad()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    dev_acc, dev_f1, *_ = model_eval_paraphrase(
      para_dev_dataloader, model, device, threshold=0.5, bidirectional=False,
      prompt_template=args.prompt_template, max_length=args.max_length
    )

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, args.filepath, {
        'best_dev_acc': best_dev_acc,
        'best_epoch': epoch,
        'prompt_template': args.prompt_template,
      })

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}, dev f1 :: {dev_f1 :.3f}")
  record_experiment(args, 'train_dev', {'dev_acc': best_dev_acc})
  return best_dev_acc


@torch.no_grad()
def load_model_for_prediction(args, device):
  saved = load_checkpoint(args.filepath)
  saved_args = saved['args']
  saved_args = add_arguments(saved_args)
  if not hasattr(saved_args, 'yes_token_id') or not hasattr(saved_args, 'no_token_id'):
    saved_args = verify_yes_no_tokens(saved_args)

  model = ParaphraseGPT(saved_args)
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  if getattr(args, 'prompt_template_was_default', False):
    args.prompt_template = getattr(saved_args, 'prompt_template', 'baseline')
  if args.max_length is None:
    args.max_length = getattr(saved_args, 'max_length', None)
  print(f"Loaded model from {args.filepath}")
  return model


@torch.no_grad()
def predict_dev(args):
  """Evaluate your model on the dev dataset; save dev predictions to disk."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  model = load_model_for_prediction(args, device)

  para_dev_data = maybe_limit_data(load_paraphrase_data(args.para_dev), args.max_dev_examples)

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_dev_dataloader = build_paraphrase_dataloader(para_dev_data, args.batch_size, shuffle=False)

  dev_para_acc, dev_para_f1, dev_para_y_pred, _, dev_para_sent_ids, _ = model_eval_paraphrase(
    para_dev_dataloader, model, device, threshold=args.threshold, bidirectional=args.bidirectional,
    prompt_template=args.prompt_template, max_length=args.max_length
  )
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}, f1 :: {dev_para_f1 :.3f}")
  write_paraphrase_predictions(args.para_dev_out, dev_para_sent_ids, dev_para_y_pred)
  record_experiment(args, 'dev_predict', {'dev_acc': dev_para_acc, 'dev_f1': dev_para_f1})


@torch.no_grad()
def calibrate_dev(args):
  """Find the best yes-probability threshold on dev only."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  model = load_model_for_prediction(args, device)

  para_dev_data = maybe_limit_data(load_paraphrase_data(args.para_dev), args.max_dev_examples)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_dev_dataloader = build_paraphrase_dataloader(para_dev_data, args.batch_size, shuffle=False)

  _, _, _, y_true, _, yes_probs = model_eval_paraphrase(
    para_dev_dataloader, model, device, threshold=0.5, bidirectional=args.bidirectional,
    prompt_template=args.prompt_template, max_length=args.max_length
  )
  threshold, acc = best_threshold_for_accuracy(
    y_true, yes_probs, args.threshold_min, args.threshold_max, args.threshold_step
  )
  calibrated_preds = [int(prob >= threshold) for prob in yes_probs]
  f1 = 0.0
  try:
    from sklearn.metrics import f1_score
    f1 = f1_score(y_true, calibrated_preds, average='macro')
  except Exception:
    pass
  print(f"best dev threshold :: {threshold:.4f}, dev acc :: {acc:.3f}, dev f1 :: {f1:.3f}")
  record_experiment(args, 'calibrate_dev', {
    'dev_acc': acc,
    'dev_f1': f1,
    'selected_threshold': threshold,
  })


@torch.no_grad()
def predict_test(args):
  """Generate final test predictions. Use only after dev settings are fixed."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  model = load_model_for_prediction(args, device)

  para_test_data = load_paraphrase_data(args.para_test, split='test')
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)
  para_test_dataloader = build_paraphrase_dataloader(para_test_data, args.batch_size, shuffle=False)

  test_para_y_pred, test_para_sent_ids, _ = model_test_paraphrase(
    para_test_dataloader, model, device, threshold=args.threshold, bidirectional=args.bidirectional,
    prompt_template=args.prompt_template, max_length=args.max_length
  )
  write_paraphrase_predictions(args.para_test_out, test_para_sent_ids, test_para_y_pred)
  record_experiment(args, 'test_predict', {})


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--grad_accum_steps", type=int, default=1)
  parser.add_argument("--max_train_examples", type=int, default=None)
  parser.add_argument("--max_dev_examples", type=int, default=None)
  parser.add_argument("--max_length", type=int, default=None)
  parser.add_argument("--prompt_template", type=str, choices=sorted(PARAPHRASE_PROMPT_TEMPLATES), default=None)
  parser.add_argument("--mode", type=str,
                      choices=['train_dev', 'dev_predict', 'calibrate_dev', 'test_predict'],
                      default='train_dev')
  parser.add_argument("--output_tag", type=str, default=None)
  parser.add_argument("--filepath", type=str, default=None)
  parser.add_argument("--experiment_log", type=str, default="results/paraphrase_experiments.csv")
  parser.add_argument("--bidirectional", action='store_true')
  parser.add_argument("--threshold", type=float, default=0.5)
  parser.add_argument("--threshold_min", type=float, default=0.30)
  parser.add_argument("--threshold_max", type=float, default=0.70)
  parser.add_argument("--threshold_step", type=float, default=0.01)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

  args = parser.parse_args()
  return args


def finalize_args(args):
  args = add_arguments(args)
  args.prompt_template_was_default = args.prompt_template is None
  if args.prompt_template is None:
    args.prompt_template = 'baseline'
  if args.grad_accum_steps < 1:
    raise ValueError('--grad_accum_steps must be >= 1')
  if args.output_tag is None:
    args.output_tag = f'{args.prompt_template}-{args.epochs}-{args.lr}'
  if args.filepath is None:
    args.filepath = os.path.join('checkpoints', f'{args.output_tag}-paraphrase.pt')
  if args.para_dev_out == "predictions/para-dev-output.csv" and args.output_tag:
    args.para_dev_out = f'predictions/para-dev-{args.output_tag}.csv'
  ensure_output_dirs(args)
  return verify_yes_no_tokens(args)


def add_arguments(args):
  """모델 크기에 따라 결정되는 인수들을 추가."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args = finalize_args(args)
  seed_everything(args.seed)  # 재현성을 위한 random seed 고정.
  if args.mode == 'train_dev':
    train(args)
    predict_dev(args)
  elif args.mode == 'dev_predict':
    predict_dev(args)
  elif args.mode == 'calibrate_dev':
    calibrate_dev(args)
  elif args.mode == 'test_predict':
    predict_test(args)
