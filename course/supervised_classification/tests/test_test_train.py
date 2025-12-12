import pandas as pd
import pytest  # noqa: F401
from course.supervised_classification.split_test_train import split_data


def test_test_and_train_creates_files(tmp_path):
    # Create dummy dataset
    df = pd.DataFrame({
        'built_age': ['Owner occupied', 'Rented', 'Owner occupied', 'Rented'],
        'feature1': [1, 2, 3, 4],
        'feature2': [5, 6, 7, 8]
    })

    # Save to temporary base_data_path
    base_data_path = tmp_path / "energy.csv"
    df.to_csv(base_data_path, index=False)

    # Define output paths
    X_train_path = tmp_path / "X_train.csv"
    y_train_path = tmp_path / "y_train.csv"
    X_test_path = tmp_path / "X_test.csv"
    y_test_path = tmp_path / "y_test.csv"

    # Run the function
    split_data(base_data_path, X_train_path, y_train_path, X_test_path, y_test_path)

    # Check that files were created
    for path in [X_train_path, y_train_path, X_test_path, y_test_path]:
        assert path.exists()

    # Check contents
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    assert not X_train.empty
    assert not y_train.empty
    assert 'feature1' in X_train.columns
    assert 'feature2' in X_train.columns
