import pytest
import pandas as pd
from plotly.graph_objects import Figure
from course.regression.eda import _boxplot


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Category': ['A', 'A', 'B', 'B'],
        'Value': [10, 15, 20, 25]
    })


def test_boxplot_returns_figure(sample_data):
    fig = _boxplot(sample_data, 'Category', 'Value', 'Test Title')
    assert isinstance(fig, Figure)


def test_boxplot_title(sample_data):
    fig = _boxplot(sample_data, 'Category', 'Value', 'Test Title')
    assert fig.layout.title.text == 'Test Title'


def test_boxplot_axes(sample_data):
    fig = _boxplot(sample_data, 'Category', 'Value', 'Test Title')
    assert fig.data[0].x is not None
    assert fig.data[0].y is not None
