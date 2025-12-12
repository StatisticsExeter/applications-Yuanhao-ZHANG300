import pandas as pd
import plotly.express as px
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path("data_cache") / "vignettes" / "unsupervised_classification"


def plot_scatter():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / "data_cache" / "la_collision.csv")
    outpath = base_dir / VIGNETTE_DIR / "scatterplot.html"
    title = "Crash types in each Local Authority"
    fig = _scatter(df, title)
    fig.write_html(outpath)


def _scatter(df: pd.DataFrame, title: str):
    """When called with dataframe `df` and a string `title`
    return a plotly express scatter_matrix of all numeric variables
    in the dataframe. The title should be as provided in the function call.
    """
    numeric_df = df.select_dtypes(include="number")
    fig = px.scatter_matrix(numeric_df, title=title)
    return fig
