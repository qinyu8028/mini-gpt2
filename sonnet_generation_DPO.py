"""
log_pi_theta_w  = f(policy_model,  prompt + chosen)
log_pi_theta_l  = f(policy_model,  prompt + rejected)
log_pi_ref_w    = f(ref_model,     prompt + chosen)     # no_grad
log_pi_ref_l    = f(ref_model,     prompt + rejected)   # no_grad

loss = -log_sigmoid(β * ((log_pi_theta_w - log_pi_ref_w) - (log_pi_theta_l - log_pi_ref_l)))
"""

import argparse
import random
import os

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
  DPOSonnetDataset,
  SonnetsDataset,
)
from sonnet_generation import SonnetGPT, save_model
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


def compute_log_prob(model: SonnetGPT, input_token_ids, attention_mask, x_len):
  logits = model(input_token_ids, attention_mask)
  prob = F.log_softmax(logits, dim=-1)

  prob = prob[:, :-1, :]  # shape: (batch, seq_len-1, vocab_size)
  input_token_ids = input_token_ids[:, 1:] # shape: (batch, seq_len-1)
  per_token_prob = prob.gather(dim=-1, index=input_token_ids.unsqueeze(-1)).squeeze(-1) # shape: (batch, seq_len-1)

  positions = torch.arange(per_token_prob.shape[1], device=per_token_prob.device)
  x_mask = (positions >= (x_len - 1).unsqueeze(1)).float() * attention_mask[:, 1:].float()
  per_token_prob_masked = per_token_prob * x_mask

  total = per_token_prob_masked.sum(dim=-1)
  
  return total 


def DPO_loss(ref_model, policy_model, batch, args):
  # loss = -log_sigmoid(β * ((log_pi_theta_w - log_pi_ref_w) - (log_pi_theta_l - log_pi_ref_l)))
  policy_win = compute_log_prob(policy_model, 
                                batch['positive_token_ids'], 
                                batch['positive_attention_mask'], 
                                batch['x_len'])
  policy_lose = compute_log_prob(policy_model, 
                                batch['negative_token_ids'], 
                                batch['negative_attention_mask'], 
                                batch['x_len'])

  with torch.no_grad():
    ref_win = compute_log_prob(ref_model, 
                                batch['positive_token_ids'],
                                batch['positive_attention_mask'], 
                                batch['x_len'])
    ref_lose = compute_log_prob(ref_model, 
                                batch['negative_token_ids'], 
                                batch['negative_attention_mask'], 
                                batch['x_len'])

  loss = - F.logsigmoid(args.beta * ((policy_win - ref_win) - (policy_lose - ref_lose))).mean()

  return loss


def train(args):
  device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
  os.makedirs('checkpoints/dpo', exist_ok=True)
  DPO_dataset = DPOSonnetDataset(positive_path=args.sonnet_path, negative_path=args.negative_out)
  DPO_dataloader = DataLoader(DPO_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=DPO_dataset.collate_fn)
  
  args = add_arguments(args)
  saved = torch.load(args.save_path, weights_only=False)

  policy_model = SonnetGPT(saved['args'])
  policy_model.load_state_dict(saved['model'], strict=False if args.use_lora else True)
  policy_model = policy_model.to(device)

  ref_model = SonnetGPT(saved['args'])
  ref_model.load_state_dict(saved['model'], strict=False if args.use_lora else True)
  ref_model = ref_model.to(device)
  ref_model.eval()
  for param in ref_model.parameters():
    param.requires_grad = False

  optimizer = AdamW(policy_model.parameters(), lr=args.lr)
  best_loss = float('inf')

  for epoch in range(args.epochs):
    policy_model.train()
    train_loss = 0
    num_batches = 0
    
    for batch in tqdm(DPO_dataloader, desc=f'DPO-train-{epoch}', disable=TQDM_DISABLE):
      batch = {k: v.to(device) for k, v in batch.items()}
      
      optimizer.zero_grad()
      loss = DPO_loss(ref_model=ref_model, policy_model=policy_model, batch=batch, args=args)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1
    
    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss: {train_loss: .3f}.")

    if train_loss < best_loss:
      best_loss = train_loss  # Save best model
      save_model(policy_model, optimizer, args, args.dpo_path)  # Save path.
      print(f"Best model saved. Best loss: {best_loss: .3f}.")

    if (epoch + 1) % 5 == 0:
      base = args.dpo_path.replace('.pt', f'_epoch{epoch}.pt')
      save_model(policy_model, optimizer, args, base)
      print(f"Saving model to {base}...")


@torch.no_grad()
def generate_dpo_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.dpo_path, weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'], strict=False if args.use_lora else True)
  model = model.to(device)
  model.eval()

  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets_dpo.txt")
  parser.add_argument("--negative_out", type=str, default="predictions/negative_sonnets.txt")
  parser.add_argument("--save_path", type=str, default="checkpoints/best_sonnet.pt")
  parser.add_argument("--dpo_path", type=str, default="checkpoints/dpo/dpo_best_sonnet.pt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=0.8)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)
  parser.add_argument("--repetition_penalty", type=float, default=1.05)
  parser.add_argument("--repetition_window", type=int, default=20)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=4)
  parser.add_argument("--lr", type=float, help="learning rate", default=5e-6)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  parser.add_argument("--use_lora", action='store_true', default=False)
  parser.add_argument("--r", help='r for LoRA', type=int, default=8)
  parser.add_argument("--alpha", type=float, 
                      help='alpha for LoRA, recommend to initialize α/r = 1 and finetune r', default=8)

  parser.add_argument("--beta", type=float, help='beta for DPO', default=0.1)

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
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


if __name__ == '__main__':
  args = get_args()
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  if args.use_lora:
    args.save_path = args.save_path.replace('.pt', f'-lora-r{args.r}.pt')
    args.dpo_path = args.dpo_path.replace('.pt', f'-lora-r{args.r}.pt')
  train(args)
  generate_dpo_sonnets(args)