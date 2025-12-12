import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm
import plotly.express as px


def plot_scatter(df: pd.DataFrame, x_name: str, y_name: str):
    """
    Given a dataframe containing numeric columns specified by x_name and y_name
    return a plotly express scatterplot.
    """
    fig = px.scatter(
        df,
        x=x_name,
        y=y_name,
        title=f"Scatter of {y_name} against {x_name}",
    )
    return fig


def calculate_correlation(df: pd.DataFrame, x1: str, x2: str):
    """
    Given a dataframe containing numeric columns specified by x1 and x2
    return two numbers:
      - Pearson correlation coefficient
      - p-value (significance of this estimate)
    """
    r, p_value = pearsonr(df[x1], df[x2])
    return r, p_value


def fit_regression(df: pd.DataFrame, x_name: str, y_name: str):
    """
    Given a dataframe containing numeric columns specified by x_name and y_name
    return the statsmodels OLS fit of a regression model of y on x.
    """
    X = sm.add_constant(df[x_name])  # 添加截距
    y = df[y_name]
    model = sm.OLS(y, X).fit()
    return model


def filter_data(df: pd.DataFrame, year: int):
    """
    Given a dataframe of various rows including a column 'Year' and an integer
    year, return a dataframe containing only those rows where the value in this
    column is less than the value of the supplied year.
    """
    return df[df["Year"] < year]


def tyler_viglen():
    """
    Create the data used at
    https://www.tylervigen.com/spurious/correlation/19524_kerosene-used-in-india_correlates-with_the-divorce-rate-in-maine
    """
    array_1 = np.array([
        227, 238.806, 220.93, 220.358, 216.652, 198.424, 198.587, 201.298,
        198.333, 196.481, 197.041, 189.078, 183, 166, 157, 154, 149, 128,
        88, 76.9508, 59.2101, 40.4265, 34.1946,
    ])
    array_2 = np.array([
        5.1, 5, 4.7, 4.6, 4.4, 4.3, 4.1, 4.2, 4.2, 4.2, 4.1, 4.2, 4.2, 3.9,
        3.96973, 3.58172, 3.42805, 3.42852, 3.22627, 3.19709, 3.033,
        2.40567, 2.72837,
    ])
    years = np.arange(1999, 2022)
    return pd.DataFrame(
        {"Year": years, "Kerosene": array_1, "DivorceRate": array_2}
    )
