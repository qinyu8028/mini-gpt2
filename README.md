# mini-gpt2

A from-scratch GPT-2 implementation for the Stanford CS224N final project, covering multi-head attention, transformer layers, and AdamW optimizer. The model is applied to sentiment classification (SST / CFIMDB), paraphrase detection (Quora), and Shakespearean sonnet generation.

Additionally, this repo implements **LoRA** for parameter-efficient fine-tuning and **DPO** (Direct Preference Optimization) for improving sonnet generation quality.

## Getting Started

```bash
conda env create -f env.yml
conda activate mini-gpt2
```

## Model Architecture

GPT-2 is a decoder-only transformer. Our implementation (`gpt2` size) consists of:

- **Tokenizer**: Byte Pair Encoding (BPE), vocab size 50257
- **Embeddings**: token embedding + learnable positional embedding, max sequence length 1024
- **Transformer layers**: 12 layers, each with masked multi-head self-attention (12 heads), layer norm, and MLP (768 → 3072 → 768)
- **Hidden dimension**: 768
- **Dropout**: 0.1 (after attention, MLP, and embeddings)
- **Output**: next-token prediction via language model head (tied with token embeddings)

<p align="center">
  <img src="images/transformer.png" width="350">
</p>

## Downstream Tasks

### Sentiment Classification

Fine-tune GPT-2 on SST and CFIMDB datasets using the last token representation as the sentence embedding.

```bash
python classifier.py --use_gpu --epochs 5 --lr 1e-5
```

### Paraphrase Detection

Cloze-style paraphrase detection on Quora Question Pairs.

```bash
python paraphrase_detection.py --use_gpu --epochs 1 --lr 1e-5 --batch_size 16
```

### Sonnet Generation (SFT)

Fine-tune GPT-2 on 143 Shakespeare sonnets with causal language modeling.

```bash
python sonnet_generation.py --use_gpu --epochs 60 --lr 1e-5
```

## Extensions

### LoRA

Low-Rank Adaptation: freeze pretrained weights, inject trainable low-rank matrices (W_a, W_b) into attention layers. Only ~0.3% of parameters are trained.

```bash
python sonnet_generation.py --use_gpu --epochs 20 --lr 1e-5 --use_lora --r 8 --alpha 8
python paraphrase_detection.py --use_gpu --epochs 1 --lr 1e-5 --use_lora --r 8 --alpha 8
```

### DPO (Direct Preference Optimization)

Align the model toward preferred outputs without a reward model. Uses pairs of (chosen, rejected) completions to optimize a preference loss with a KL-divergence constraint from the reference model.

**Step 1**: Generate negative samples from the SFT model:
```bash
python sonnet_generation_negative.py --use_gpu --save_path checkpoints/best_sonnet_full_e60.pt
```

**Step 2**: DPO training:
```bash
python sonnet_generation_DPO.py --use_gpu \
  --save_path checkpoints/best_sonnet_full_e60.pt \
  --dpo_path checkpoints/dpo/dpo_best.pt \
  --lr 1e-6 --beta 0.1 --epochs 10
```

## Results (Sonnet Generation)

| Method | Config | Dev chrF |
|--------|--------|----------|
| SFT (full-model) | 60 epochs | 40.34 |
| DPO | beta=0.1, lr=1e-6, epoch 4 | 42.23 |

DPO provides modest improvement over SFT. Note that chrF is not very sensitive to qualitative improvements in this task.

## Acknowledgement

Based on starter code from [Stanford CS224N (Winter 2026) Default Final Project](https://github.com/stanfordnlp/cs224n_gpt).

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
