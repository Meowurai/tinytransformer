# TinyTransformer

A learning lab for building a minimal transformer from scratch.

The goal is to understand why the transformer architecture exists by starting from the limitations of TinyLM: fixed-size flattened contexts, limited sequence structure, and no token-to-token communication.

## Status

In progress.

## Outcome

Build a transformer one architectural idea at a time, understanding the motivation for each component before implementing it.

## Progress

- ✓ Stage 1 — Input Representation
- ☐ Stage 2 — Self-Attention
- ☐ Stage 3 — Multi-Head Attention
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