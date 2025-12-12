import pytest
import pandas as pd
import plotly.graph_objects as go
from course.regression.caterpillar_reffs import _plot_caterpillar


@pytest.fixture
def sample_re_df():
    return pd.DataFrame({
        'Intercept': [0.5, -0.2, 0.1],
        'Slope_0': [0.1, 0.2, 0.3],
        'group': ['A', 'B', 'C'],
        'lower': [0.3, -0.4, -0.1],
        'upper': [0.7, 0.0, 0.3]
    })


def test_plot_caterpillar_returns_figure(sample_re_df):
    fig = _plot_caterpillar(sample_re_df, 'Intercept', 'Test Caterpillar')
    assert isinstance(fig, go.Figure)


def test_plot_caterpillar_trace_count(sample_re_df):
    fig = _plot_caterpillar(sample_re_df, 'Intercept', 'Test Caterpillar')
    # 3 CI lines + 1 scatter for point estimates
    assert len(fig.data) == 4


def test_plot_caterpillar_layout(sample_re_df):
    fig = _plot_caterpillar(sample_re_df, 'Intercept', 'Test Caterpillar')
    assert fig.layout.title.text == 'Test Caterpillar'
    assert fig.layout.xaxis.title.text == 'Effect Size'
    assert fig.layout.yaxis.title.text == 'Local Authority'


def test_plot_caterpillar_reference_line(sample_re_df):
    fig = _plot_caterpillar(sample_re_df, 'Intercept', 'Test Caterpillar')
    shapes = fig.layout.shapes
    assert any(
        shape['type'] == 'line' and shape['x0'] == 0 and shape['x1'] == 0
        for shape in shapes
    )
