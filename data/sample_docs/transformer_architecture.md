# Transformer Architecture — A Complete Guide

## Introduction

The Transformer architecture, introduced in the landmark paper "Attention Is All You Need" by Vaswani et al. (2017), revolutionized natural language processing and subsequently computer vision, speech processing, and other domains. It replaced recurrent and convolutional architectures with a purely attention-based mechanism.

## Architecture Overview

The original Transformer follows an encoder-decoder structure:

### Encoder
The encoder processes the input sequence and produces contextual representations:
1. **Input Embedding**: Convert tokens to dense vectors (dimension d_model = 512)
2. **Positional Encoding**: Add sinusoidal position information
3. **N Encoder Layers** (N=6 in the original), each containing:
   - Multi-Head Self-Attention (8 heads)
   - Position-wise Feed-Forward Network (d_ff = 2048)
   - Layer Normalization and Residual Connections

### Decoder
The decoder generates the output sequence auto-regressively:
1. **Output Embedding + Positional Encoding**
2. **N Decoder Layers** (N=6), each containing:
   - Masked Multi-Head Self-Attention
   - Multi-Head Cross-Attention (attending to encoder output)
   - Position-wise Feed-Forward Network
   - Layer Normalization and Residual Connections

## Key Components

### Positional Encoding
Since attention is permutation-invariant, positional information must be explicitly added. The original Transformer uses sinusoidal encodings:

PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Modern variants often use learned positional embeddings (BERT, GPT) or Rotary Position Embeddings (RoPE, used in Llama and most recent models).

### Feed-Forward Network
Each encoder/decoder layer contains a position-wise FFN:
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

This applies the same two-layer network to each position independently. Modern variants use:
- GELU activation instead of ReLU
- SwiGLU (Llama 2, 3): FFN(x) = (Swish(xW_1) ⊙ xV)W_2

### Layer Normalization
Applied before each sub-layer (Pre-LN, used in GPT-2+) or after (Post-LN, original):
- Pre-LN is more stable during training
- Post-LN can achieve slightly better performance with careful tuning

## Decoder-Only Models (GPT Family)

The most successful modern LLMs use decoder-only architectures:

### GPT Series
- **GPT-1** (2018): 117M parameters, 12 layers, d_model=768
- **GPT-2** (2019): 1.5B parameters, 48 layers, d_model=1600
- **GPT-3** (2020): 175B parameters, 96 layers, d_model=12288
- **GPT-4** (2023): Estimated 1.8T parameters (mixture of experts)

### Llama Series
- **Llama 1** (2023): 7B-65B parameters, RoPE, SwiGLU, Pre-LN
- **Llama 2** (2023): Same architecture, better data and RLHF
- **Llama 3** (2024): 8B and 70B, 128K context, GQA
- **Llama 3.1** (2024): Multilingual, tool use, 405B flagship

### Training

Modern LLMs are trained in stages:
1. **Pre-training**: Next-token prediction on trillions of tokens
2. **Supervised Fine-Tuning (SFT)**: Learn to follow instructions
3. **RLHF/DPO**: Align with human preferences

Training costs:
- Llama 3 70B: ~$2M in compute
- GPT-4: Estimated $50-100M
- Training typically uses clusters of thousands of GPUs

## Encoder-Only Models (BERT Family)

### BERT
Bidirectional Encoder Representations from Transformers (Devlin et al., 2018):
- 110M (base) / 340M (large) parameters
- Pre-trained with Masked Language Modeling (MLM) and Next Sentence Prediction
- Revolutionized NLP benchmarks: GLUE, SQuAD, etc.
- Still widely used for classification, NER, and embedding generation

## Scaling Laws

Kaplan et al. (2020) and Hoffmann et al. (2022, "Chinchilla") established scaling laws:
- Performance improves predictably with compute, data, and parameters
- Chinchilla optimal: tokens ≈ 20× parameters
- A 7B model should be trained on ~140B tokens for optimal efficiency

## Modern Innovations

### Mixture of Experts (MoE)
- Route each token to a subset of "expert" FFN layers
- Mixtral 8x7B: 8 experts, 2 active per token, 47B total / 12B active
- Achieves the quality of larger dense models at lower inference cost

### Grouped Query Attention (GQA)
- Share key-value heads across multiple query heads
- Llama 3 uses GQA: 32 query heads, 8 KV heads
- Reduces memory usage by ~4x with minimal quality loss

### Context Length Extensions
- RoPE scaling: NTK-aware interpolation
- Llama 3: 128K context natively
- Ring Attention: Distribute long sequences across devices

## References

- Vaswani et al. "Attention Is All You Need" (2017)
- Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
- Radford et al. "Language Models are Unsupervised Multitask Learners" (2019)
- Touvron et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models" (2023)
- Jiang et al. "Mixtral of Experts" (2024)
