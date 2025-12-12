import pytest
from plotly.graph_objects import Figure
from course.supervised_classification.roc_curve import _plot_roc_curve


@pytest.fixture
def sample_roc_data():
    return {
        'fpr': [0.0, 0.5, 1.0],
        'tpr': [0.0, 0.7, 1.0],
        'roc_auc': 0.85
    }


def test_plot_roc_curve_returns_figure(sample_roc_data):
    fig = _plot_roc_curve(sample_roc_data, sample_roc_data)
    assert isinstance(fig, Figure)


def test_plot_roc_curve_has_three_traces(sample_roc_data):
    fig = _plot_roc_curve(sample_roc_data, sample_roc_data)
    print(len(fig.data))
    assert len(fig.data) == 3  # LDA, QDA, Random baseline


def test_trace_names(sample_roc_data):
    fig = _plot_roc_curve(sample_roc_data, sample_roc_data)
    names = [trace.name for trace in fig.data]
    assert any("LDA" in name for name in names)
    assert any("QDA" in name for name in names)
    assert any("Random" in name for name in names)
