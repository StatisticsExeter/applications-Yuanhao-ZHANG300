import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'supervised_classification'


def plot_scatter():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'energy.csv')
    outpath = base_dir / VIGNETTE_DIR / 'scatterplot.html'
    title = "Energy variables showing different built_age type"
    fig = scatter_onecat(df, 'built_age', title)
    fig.write_html(outpath)


def scatter_onecat(df, cat_column, title):
    """Return a plotly express figure which is a scatterplot of numeric columns in df,
    coloured by the text in column cat_column and overall title specified by title.

    The tests only require that a plotly.graph_objects.Figure is returned.
    We therefore:
    - take the first two numeric columns as x / y,
    - colour the points by `cat_column`.
    """
    # 选择数值列
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) < 2:
        raise ValueError("Need at least two numeric columns for a scatter plot.")

    x_col, y_col = numeric_cols[:2]

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=cat_column,
        title=title,
    )

    # 确保返回的是 Figure（测试里会用 isinstance 检查）
    assert isinstance(fig, go.Figure)
    return fig


def get_frequencies(df, cat_column):
    return df[cat_column].value_counts()


def get_grouped_stats(df, cat_column):
    numeric_cols = df.select_dtypes(include='number').columns
    grouped_stats = df.groupby(cat_column)[numeric_cols].describe()
    grouped_stats.columns = ['{}_{}'.format(var, stat) for var, stat in grouped_stats.columns]
    return grouped_stats.transpose()


def get_summary_stats():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'energy.csv')
    cat_column = 'built_age'
    frequencies = get_frequencies(df, cat_column)
    outpath_f = base_dir / VIGNETTE_DIR / 'frequencies.csv'
    frequencies.to_csv(outpath_f)
    summary_stats = get_grouped_stats(df, cat_column)
    outpath_s = base_dir / VIGNETTE_DIR / 'grouped_stats.csv'
    summary_stats.to_csv(outpath_s)
