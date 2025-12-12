import pytest
from course.exercises.exercises_1 import (
    sum_list, first_of_tuple, has_key, round_float, reverse_list,
    count_occurrences, tuples_to_dict, string_length, unique_elements, swap_dict
)


@pytest.mark.parametrize("numbers, expected", [
    ([1, 2, 3], 6),
    ([0, 0, 0], 0),
    ([-1, 1], 0),
])
def test_sum_list(numbers, expected):
    assert sum_list(numbers) == expected


@pytest.mark.parametrize("t, expected", [
    ((10, 20, 30), 10),
    ((-1, 0, 1), -1),
    (('a', 'b'), 'a'),
])
def test_first_of_tuple(t, expected):
    assert first_of_tuple(t) == expected


@pytest.mark.parametrize("d, key, expected", [
    ({'a': 1, 'b': 2}, 'b', True),
    ({'a': 1}, 'z', False),
    ({}, 'a', False),
])
def test_has_key(d, key, expected):
    assert has_key(d, key) is expected


@pytest.mark.parametrize("f, expected", [
    (3.14159, 3.14),
    (2.71828, 2.72),
    (0.0, 0.0),
])
def test_round_float(f, expected):
    assert round_float(f) == expected


@pytest.mark.parametrize("lst, expected", [
    ([1, 2, 3], [3, 2, 1]),
    ([], []),
    (['a', 'b'], ['b', 'a']),
])
def test_reverse_list(lst, expected):
    assert reverse_list(lst) == expected


@pytest.mark.parametrize("lst, item, expected", [
    (['a', 'b', 'a'], 'a', 2),
    ([1, 2, 3, 1], 1, 2),
    ([], 'x', 0),
])
def test_count_occurrences(lst, item, expected):
    assert count_occurrences(lst, item) == expected


@pytest.mark.parametrize("pairs, expected", [
    ([('a', 1), ('b', 2)], {'a': 1, 'b': 2}),
    ([], {}),
    ([('x', 42)], {'x': 42}),
])
def test_tuples_to_dict(pairs, expected):
    assert tuples_to_dict(pairs) == expected


@pytest.mark.parametrize("s, expected", [
    ("hello", 5),
    ("", 0),
    ("Python", 6),
])
def test_string_length(s, expected):
    assert string_length(s) == expected


@pytest.mark.parametrize("lst, expected", [
    ([1, 2, 2, 3], [1, 2, 3]),
    (['a', 'a', 'b'], ['a', 'b']),
    ([], []),
])
def test_unique_elements(lst, expected):
    assert sorted(unique_elements(lst)) == sorted(expected)


@pytest.mark.parametrize("d, expected", [
    ({'a': 1, 'b': 2}, {1: 'a', 2: 'b'}),
    ({'x': 42}, {42: 'x'}),
    ({}, {}),
])
def test_swap_dict(d, expected):
    assert swap_dict(d) == expected
