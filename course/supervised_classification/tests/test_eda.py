import pandas as pd
import pytest
from course.supervised_classification.eda import scatter_onecat, get_grouped_stats, get_frequencies
import plotly.graph_objects as go


def test_scatter_returns_figure():
    # Create a temporary CSV file
    data = {
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'feature3': [7, 8, 9],
        'category': ['A', 'B', 'A']
    }
    df = pd.DataFrame(data)

    # Call the function
    fig = scatter_onecat(df, cat_column='category', title='boo')

    # Assertions
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # There should be some traces


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'built_age': ['Owned', 'Rented', 'Owned', 'Rented', 'Shared'],
        'energy_score': [70, 55, 80, 60, 65],
        'floor_area': [100, 80, 120, 85, 90]
    })


def test_frequencies(sample_df):
    freq = get_frequencies(sample_df, 'built_age')
    assert freq['Owned'] == 2
    assert freq['Rented'] == 2
    assert freq['Shared'] == 1
    assert freq.sum() == len(sample_df)


def test_grouped_stats(sample_df):
    stats = get_grouped_stats(sample_df, 'built_age')
    print(stats.columns)
    print(stats.index)
    assert 'Owned' in stats.columns
    assert 'energy_score_count' in stats.index
