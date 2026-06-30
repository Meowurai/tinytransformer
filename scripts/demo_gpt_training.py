import random

from pathlib import Path 

from tinytransformer.dataset import build_example, count_examples
from tinytransformer.loss import cross_entropy_loss
from tinytransformer.tokenizer import Tokenizer
from tinytransformer.training import zero_grad, step
from tinytransformer.transformer import Transformer
from tinytransformer.activations import softmax

DATA_PATH = Path("data/tinyshakespeare.txt")

def sample_from_logits(logits):
    probabilities = softmax(logits)

    r = random.random()
    cumulative = 0.0

    for index, probability in enumerate(probabilities):
        cumulative += probability.data
        if r <= cumulative:
            return index

    return len(probabilities) - 1

def generate(model, start_ids, max_new_tokens, block_size):
    ids = list(start_ids)

    for _ in range(max_new_tokens):
        context = ids[-block_size:]
        logits = model(context)

        next_token_logits = logits[-1]
        next_token_id = sample_from_logits(next_token_logits)

        ids.append(next_token_id)

    return ids

def main():
    text = DATA_PATH.read_text()
    tokenizer = Tokenizer.from_text(text)
    token_ids = tokenizer.encode(text)

    block_size = 8

    model = Transformer(
        vocabulary_size=tokenizer.vocabulary_size,
        embedding_size=8,
        num_heads=2,
        num_blocks=1,
        block_size=block_size
    )

    example_count = count_examples(token_ids, block_size)
    max_steps = 1000
    for step_index in range(max_steps):
        start_index = random.randint(0, example_count -1)
        x, y = build_example(
            token_ids,
            start_index=start_index,
            block_size=block_size
        )

        logits = model(x)
        loss = cross_entropy_loss(logits, y)

        zero_grad(model)
        loss.backward()
        step(model, learning_rate=0.01)

        if step_index % 100 == 0:
            print("step:", step_index, "loss:", loss.data)

    start_text = "To be"
    start_ids = tokenizer.encode(start_text)

    generated_ids = generate(
        model=model,
        start_ids=start_ids,
        max_new_tokens=50,
        block_size=block_size
    )

    print()
    print("Generated text:")
    print(tokenizer.decode(generated_ids))

if __name__ == "__main__":
    main()