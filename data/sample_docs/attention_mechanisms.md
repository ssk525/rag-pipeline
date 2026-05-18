# Attention Mechanism in Neural Networks

## Introduction

The attention mechanism is a fundamental component in modern deep learning architectures. It was first introduced in the context of neural machine translation by Bahdanau et al. (2014), allowing models to focus on different parts of the input sequence when generating each element of the output.

## How Attention Works

The core idea behind attention is to compute a weighted sum of values, where the weights (attention scores) are determined by the compatibility between a query and a set of keys.

### Scaled Dot-Product Attention

The most common form of attention used in Transformers is scaled dot-product attention:

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

Where:
- Q (Query): The current element we're focusing from
- K (Key): Elements we're comparing against
- V (Value): The actual information we want to aggregate
- d_k: The dimension of the keys (used for scaling)

The scaling factor 1/sqrt(d_k) prevents the dot products from growing too large, which would push the softmax into regions with extremely small gradients.

## Multi-Head Attention

Rather than performing a single attention function, multi-head attention allows the model to jointly attend to information from different representation subspaces:

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

Where each head is computed as:
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

This allows different attention heads to capture different types of relationships in the data, such as syntactic dependencies, semantic similarities, or positional patterns.

## Self-Attention vs Cross-Attention

### Self-Attention
In self-attention, the queries, keys, and values all come from the same sequence. This is used in:
- Transformer encoders (BERT, GPT)
- Understanding relationships within a single input

### Cross-Attention
In cross-attention, queries come from one sequence while keys and values come from another. This is used in:
- Encoder-decoder models (translation, summarization)
- Connecting different modalities (text-to-image)

## Attention in Computer Vision

### Vision Transformers (ViT)
Vision Transformers divide images into patches and apply self-attention to these patches. This allows the model to capture long-range dependencies in images that convolutional neural networks struggle with.

The ViT architecture:
1. Split image into fixed-size patches (e.g., 16x16 pixels)
2. Flatten and linearly project each patch
3. Add positional embeddings
4. Feed through Transformer encoder layers
5. Use [CLS] token output for classification

### Performance
ViT-Large achieves 87.76% top-1 accuracy on ImageNet when pre-trained on large datasets, competitive with the best convolutional networks.

## Computational Complexity

Standard self-attention has O(n^2) complexity with respect to sequence length n, which becomes a bottleneck for long sequences. Various approaches have been proposed to address this:

- **Sparse Attention**: Attend only to a subset of positions (e.g., Longformer)
- **Linear Attention**: Approximate attention with linear complexity (e.g., Performer)
- **Flash Attention**: Hardware-aware optimization that reduces memory usage from O(n^2) to O(n)

Flash Attention achieves 2-4x speedup on common sequence lengths (512-2048) while being mathematically equivalent to standard attention.

## Applications

Attention mechanisms are now used across virtually all areas of AI:
- **NLP**: Machine translation, text generation, question answering
- **Computer Vision**: Image classification, object detection, segmentation
- **Speech**: Speech recognition, text-to-speech synthesis
- **Multimodal**: CLIP, DALL-E, GPT-4V combine text and image attention

## References

- Vaswani et al. "Attention Is All You Need" (2017)
- Bahdanau et al. "Neural Machine Translation by Jointly Learning to Align and Translate" (2014)
- Dosovitskiy et al. "An Image is Worth 16x16 Words" (2020)
- Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)
