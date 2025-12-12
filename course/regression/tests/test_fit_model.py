import pytest
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLMResultsWrapper
from course.regression.fit_model import _fit_model, _random_effects


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'shortfall': [10, 12, 9, 14, 13, 11],
        'n_rooms': [3, 4, 2, 5, 4, 3],
        'age': [30, 45, 25, 50, 40, 35],
        'local_authority_code': ['A', 'A', 'B', 'B', 'C', 'C']
    })


def test_fit_model_returns_results(sample_df):
    results = _fit_model(sample_df)
    print(type(results))
    assert isinstance(results, MixedLMResultsWrapper)


def test_random_effects_structure(sample_df):
    results = _fit_model(sample_df)
    re_df = _random_effects(results)

    # Check expected columns
    slopes = [col for col in re_df.columns if col.startswith('Slope_')]
    expected_cols = ['Intercept'] + slopes + ['group', 'lower', 'upper']
    for col in expected_cols:
        assert col in re_df.columns

    # Check that 'group' matches index
    assert all(re_df['group'] == re_df.index)

    # Check that 'lower' and 'upper' bounds are computed correctly
    stderr = np.sqrt(results.cov_re.iloc[0, 0])
    expected_lower = re_df['Intercept'] - 1.96 * stderr
    expected_upper = re_df['Intercept'] + 1.96 * stderr
    assert np.allclose(re_df['lower'], expected_lower)
    assert np.allclose(re_df['upper'], expected_upper)
