import pytest
import pandas as pd
from ..pipeline_functions import tyler_viglen, calculate_correlation, filter_data, fit_regression


def test_tyler_viglen():
    df = tyler_viglen()
    assert len(df) == 23
    assert isinstance(df, pd.DataFrame)


def test_filter_data():
    df = pd.DataFrame({
        'Year': [2010, 2012, 2015, 2018],
        'Value': [1, 2, 3, 4]
    })
    filtered = filter_data(df, 2015)
    assert all(filtered['Year'] < 2015)
    assert len(filtered) == 2


@pytest.mark.parametrize("input_df, expected_r", [
    (pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2, 4, 6, 8, 10]
    }), 1),
])
def test_calculate_correlation_basic(input_df, expected_r):
    corr, p = calculate_correlation(input_df, 'x', 'y')
    assert round(corr, 2) == expected_r


def test_fit_regression_basic():
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2, 4, 6, 8, 10]
    })
    model = fit_regression(df, 'x', 'y')
    assert round(model.params['x'], 2) == 2.0
    assert round(model.rsquared, 2) == 1.0


def test_fit_regression_model_type():
    df = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [2, 4, 6]
    })
    model = fit_regression(df, 'x', 'y')
    assert hasattr(model, 'params')
    assert hasattr(model, 'rsquared')
