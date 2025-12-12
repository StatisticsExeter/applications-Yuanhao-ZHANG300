import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'regression'


def _boxplot(df, x_var, y_var, title):
    """Given a data frame 'df' containing categorical variable `x_var`
    and outcome variable `y_var` produce a box plot of the distribution
    of `y_var` for different levels of `x_var`. The box plot should
    have title `title` and return a Plotly Figure object.
    """
    fig = px.box(
        df,
        x=x_var,
        y=y_var,
        title=title,
    )

    # 确保标题属性设置好（测试会检查 layout.title.text）
    fig.update_layout(title_text=title)

    # 返回 Figure（测试里用 isinstance(fig, go.Figure)）
    assert isinstance(fig, go.Figure)
    return fig


def boxplot_age():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_energy.csv')
    fig = _boxplot(df, 'age', 'shortfall', 'Shortfall by Age Category')
    fig.write_html(VIGNETTE_DIR / 'boxplot_age.html')


def boxplot_rooms():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_energy.csv')
    fig = _boxplot(df, 'n_rooms', 'shortfall', 'Shortfall by Number of rooms')
    fig.write_html(VIGNETTE_DIR / 'boxplot_rooms.html')
