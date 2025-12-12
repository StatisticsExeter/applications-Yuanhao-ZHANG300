# python_exercises.py
from datetime import datetime

def sum_list(lst):
    """Return the sum of a list. Empty list -> 0."""
    return sum(lst)


def max_value(lst):
    """Return the maximum value in a list."""
    return max(lst)


def reverse_string(s):
    """Reverse a string."""
    return s[::-1]


def filter_even(lst):
    """Return a list of even numbers."""
    return [x for x in lst if x % 2 == 0]


def get_fifth_row(df):
    """Return the 5th row (index 4). Raise IndexError if too short."""
    if len(df) < 5:
        raise IndexError("DataFrame has fewer than 5 rows")
    return df.iloc[4]


def column_mean(df, column):
    """Return the mean of df[column]. If column empty -> NaN."""
    if column not in df.columns:
        raise KeyError(f"Column {column} not found")
    if len(df[column]) == 0:
        return float('nan')
    return df[column].mean()


def lookup_key(d, key):
    """Return dict[key] or None if missing."""
    return d.get(key, None)


def count_occurrences(lst):
    """Count occurrences of items in list and return dict."""
    counts = {}
    for item in lst:
        counts[item] = counts.get(item, 0) + 1
    return counts


def list_to_string(lst):
    """Join a list into a comma-separated string."""
    return ",".join(lst)


def parse_date(date_str):
    """Parse 'YYYY-MM-DD' into datetime.date."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.date()
    except Exception:
        raise ValueError("Invalid date format")

def greet():
    """
    Prints 'Hello, World!' exactly as required by the tests.
    """
    print("Hello, World!")


def add_two_numbers(a, b):
    """
    Returns the sum of a and b.
    """
    return a + b
