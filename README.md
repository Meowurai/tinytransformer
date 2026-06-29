# TinyTransformer

A learning lab for building a minimal transformer from scratch.

The goal is to understand why the transformer architecture exists by starting from the limitations of TinyLM: fixed-size flattened contexts, limited sequence structure, and no token-to-token communication.

## Status

In progress.

## Outcome

Build a transformer one architectural idea at a time, understanding the motivation for each component before implementing it.

## Progress

- ✓ Stage 1 — Input Representation
- ✓ Stage 2 — Self-Attention
- ✓ Stage 3 — Multi-Head Attention
- ☐ Stage 4 — Transformer Block
- ☐ Stage 5 — Stacking Blocks
- ☐ Stage 6 — Character-Level GPT

## Stage 1: Input Representation

The first stage builds the complete input representation required by a transformer. Unlike TinyLM, the sequence structure is preserved rather than flattened.

### Architecture

```text
Raw Text
    │
    ▼
Character Tokenizer
    │
    ▼
Token IDs
    │
    ▼
Sequence Dataset
    │
    ▼
Token Embeddings
          +
Position Embeddings
    │
    ▼
Input Embeddings
```

### Components

- **Tokenizer**
  - Builds a character-level vocabulary from the corpus.
  - Encodes and decodes text.
- **Dataset**
  - Produces shifted `(input_ids, target_ids)` sequence pairs.
  - Preserves sequence structure.
- **EmbeddingTable**
  - Stores learned vectors indexed by integer IDs.
  - Used for both token and positional embeddings.

### Key Concepts Learned

- Transformers train on sequences rather than individual context/target pairs.
- Every token receives both a token embedding and a positional embedding.
- Position is represented by a learned embedding vector.
- Token and positional embeddings are added to form a single input representation.
- The sequence remains a list of vectors rather than being flattened.

### Limitation Discovered

Although every token now knows what it is and where it is, tokens are still independent. They cannot yet exchange information with one another.

## Stage 2: Self-Attention

The second stage introduces the core idea behind the transformer: every token updates its representation by attending to every other token in the sequence.

Rather than keeping a fixed embedding throughout the model, each token continuously builds a contextual representation based on the information contained in the surrounding tokens.

### Architecture

```text
Input Embeddings
      │
      ▼
Query Projection
Key Projection
Value Projection
      │
      ▼
Attention Scores
      │
      ▼
Softmax
      │
      ▼
Attention Weights
      │
      ▼
Weighted Sum of Values
      │
      ▼
Contextualized Token Representations
```

### Components

- **Linear**
  - Produces learned Query, Key, and Value projections.
- **Attention Scores**
  - Computes a relevance score between one query and every key using the dot product.
- **Softmax**
  - Converts raw attention scores into attention weights that sum to one.
- **Weighted Sum**
  - Produces a new contextual representation by combining value vectors according to the learned attention weights.
- **AttentionHead**
  - Updates every token representation by repeating this process for each token in the sequence.

### Key Concepts Learned

- Token representations are no longer fixed after the embedding layer.
- Every token compares itself to every other token in the sequence.
- Queries decide what a token is looking for.
- Keys decide how relevant each token is.
- Values contain the information contributed to other tokens.
- Self-attention creates contextual token representations by combining information from the entire sequence.

### Limitation Discovered

A single attention head can only learn one way of relating tokens. The next stage introduces multiple attention heads operating in parallel to capture different relationships within the same sequence.

## Stage 3: Multi-Head Attention

The third stage extends self-attention by running multiple independent attention heads in parallel. Rather than relying on a single way of relating tokens, the model learns several different perspectives of the same sequence before combining them into a richer representation.

### Architecture

```text
Input Embeddings
        │
        ▼
┌───────────────────────────────┐
│      Attention Head 1         │
├───────────────────────────────┤
│      Attention Head 2         │
├───────────────────────────────┤
│             ...               │
├───────────────────────────────┤
│      Attention Head N         │
└───────────────────────────────┘
        │
        ▼
Concatenate Head Outputs
        │
        ▼
Output Projection (Linear)
        │
        ▼
Contextualized Token Representations
```

### Components

- **MultiHeadAttention**
  - Owns multiple independent attention heads.
  - Concatenates their outputs before applying a final linear projection.
- **AttentionHead**
  - Learns one independent way of relating tokens.
- **Output Projection**
  - Learns how to combine the different perspectives produced by each attention head.

### Key Concepts Learned

- One attention head represents one learned perspective on a sequence.
- Multiple heads operate independently on the same input.
- Head outputs are concatenated rather than added.
- A final linear layer learns how to combine information from all heads.
- Multi-head attention increases expressiveness without changing the external embedding size.

### Limitation Discovered

Although tokens can now exchange information through multiple attention heads, the model still lacks the mechanisms needed to train very deep networks reliably. The next stage introduces residual connections, layer normalization, and the feed-forward network that together form the Transformer Block.