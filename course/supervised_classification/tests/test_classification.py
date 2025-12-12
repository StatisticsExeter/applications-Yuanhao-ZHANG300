import pandas as pd
import joblib
import pytest  # noqa: F401
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from course.supervised_classification.predict import predict


def test_predict_creates_csv(tmp_path):
    # Dummy training data
    X = pd.DataFrame([[1, 2], [3, 2], [15, 16]], columns=['First', 'Second'])
    y = pd.Series(['A', 'B', 'A'])

    # Train and save model
    model = LinearDiscriminantAnalysis()
    model.fit(X, y)
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    # Save test data
    X_test = X.iloc[:2]
    X_test_path = tmp_path / "X_test.csv"
    X_test.to_csv(X_test_path, index=False)

    # Output path
    y_pred_path = tmp_path / "y_pred.csv"
    y_pred_prob_path = tmp_path / "y_pred_prob.csv"

    # Run prediction
    predict(model_path, X_test_path, y_pred_path, y_pred_prob_path)

    # Assertions
    assert y_pred_path.exists()
    df = pd.read_csv(y_pred_path)
    assert 'predicted_built_age' in df.columns
    assert len(df) == len(X_test)
