import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'regression'


def _plot_caterpillar(re_df, target, title):
    fig = go.Figure()
    # Add CI lines for each group
    for _, row in re_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['lower'], row['upper']],
            y=[row['group'], row['group']],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False
        ))
    # Add point estimates
    fig.add_trace(go.Scatter(
        x=re_df[target],
        y=re_df['group'],
        mode='markers',
        marker=dict(color='blue', size=8),
        name='Random Intercept'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Effect Size',
        yaxis_title='Local Authority',
        template='plotly_white'
    )
    fig.add_shape(
        type="line",
        x0=0, x1=0,
        y0=0, y1=len(re_df),  # span the full y-axis
        line=dict(color="red", width=2, dash="dash")
    )
    return fig


def plot_caterpillar():
    base_dir = find_project_root()
    re_df = pd.read_csv(base_dir / 'data_cache' / 'models' / 'reffs.csv')
    fig = _plot_caterpillar(re_df, 'Intercept', '95% CI for local authority random effects')
    fig.write_html(VIGNETTE_DIR / 'caterpillar.html')
