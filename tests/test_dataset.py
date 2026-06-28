
import pytest

from tinytransformer.dataset import build_example, count_examples

def test_build_example_returns_shifted_input_and_target():
    ids = [1, 2, 3, 4, 5]

    x, y = build_example(ids, start_index=0, block_size=4)

    assert x == [1, 2, 3, 4]
    assert y == [2, 3, 4, 5]

def test_build_example_uses_start_index():
    ids = [1, 2, 3, 4, 5, 6]

    x, y = build_example(ids, start_index=1, block_size=4)

    assert x == [2, 3, 4, 5]
    assert y == [3, 4, 5, 6]


def test_build_example_raises_when_window_overshoots():
    ids = [1, 2, 3, 4, 5]

    with pytest.raises(ValueError):
        build_example(ids, start_index=1, block_size=4)

def test_count_examples_returns_zero_when_not_enough_tokens():
    assert count_examples([1, 2, 3], block_size=4) == 0