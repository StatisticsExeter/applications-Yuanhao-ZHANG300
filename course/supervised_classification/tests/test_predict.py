import pandas as pd
import joblib
import pytest  # noqa: F401
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from course.supervised_classification.predict import predict


def test_predict_creates_output(tmp_path):
    # Create dummy data
    X = pd.DataFrame([[1, 2], [3, 2], [15, 16]], columns=['First', 'Second'])
    y = pd.Series(['A', 'B', 'A'])

    # Train a simple model
    model = LinearDiscriminantAnalysis()
    model.fit(X, y)

    # Save model and test data
    model_path = tmp_path / "model.joblib"
    X_test_path = tmp_path / "X_test.csv"
    y_pred_path = tmp_path / "y_pred.csv"
    y_pred_prob_path = tmp_path / "y_pred.csv"

    joblib.dump(model, model_path)
    X.iloc[:2].to_csv(X_test_path, index=False)

    # Run prediction
    predict(model_path, X_test_path, y_pred_path, y_pred_prob_path)

    # Check output
    assert y_pred_path.exists()
    y_pred = pd.read_csv(y_pred_path)
    assert len(y_pred) == 2
    assert 'predicted_built_age' in y_pred.columns
