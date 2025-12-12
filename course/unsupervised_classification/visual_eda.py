import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from course.unsupervised_classification.eda import _scatter


def summary_stats():
    target_path = 'vignettes/unsupervised/cache/olive_oil_summary.html'
    df = pd.read_csv("data_cache/unsupervised.csv")
    df.describe().round(1).to_html(target_path, index=False)


def generate_raw_boxplot():
    df = pd.read_csv("data_cache/unsupervised.csv")
    df_melted = df.melt(var_name='Variable', value_name='Value')
    # Create and save box plot
    fig = px.box(df_melted, x='Variable', y='Value', title='Scaled Box Plot of Olive Oil Variables')
    fig.write_html("vignettes/unsupervised/cache/raw_boxplot.html")


def generate_scaled_boxplot():
    df = pd.read_csv("data_cache/unsupervised.csv")
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df_melted = df_scaled.melt(var_name='Variable', value_name='Value')
    # Create and save box plot
    fig = px.box(df_melted, x='Variable', y='Value', title='Scaled Box Plot of Olive Oil Variables')
    fig.write_html("vignettes/unsupervised/cache/scaled_boxplot.html")


def generate_scatterplot():
    df = pd.read_csv("data_cache/unsupervised.csv")
    fig = _scatter(df, title="Scatter Matrix of Continuous Variables")
    fig.write_html("vignettes/unsupervised/cache/scatterplot.html")
