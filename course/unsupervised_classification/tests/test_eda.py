import pandas as pd
import plotly.express as px
from course.unsupervised_classification.eda import _scatter  # replace with actual module name


def test_scatter_returns_plotly_figure():
    # Create a sample DataFrame
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': [7, 8, 9]
    })
    title = "Test Scatter Matrix"

    # Call the function
    fig = _scatter(df, title)

    # Check that the result is a plotly figure
    assert isinstance(fig, px.scatter_matrix(df).__class__)

    # Check that the title is correctly set
    assert fig.layout.title.text == title

    # Check that all columns are used as dimensions
    dimension_labels = [dim.label for dim in fig.data[0].dimensions]
    assert set(dimension_labels) == set(df.columns)
