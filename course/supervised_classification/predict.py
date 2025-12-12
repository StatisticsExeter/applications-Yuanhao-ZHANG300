import joblib
import pandas as pd
from pathlib import Path
from course.utils import find_project_root


def predict(model_path, X_test_path, y_pred_path, y_pred_prob_path):
    """
    Load a classifier and test data, generate predictions, and save:
      - y_pred : predicted class labels, with column name 'predicted_built_age'
      - y_pred_prob : predicted probabilities (if available)
    """
    model_path = Path(model_path)
    X_test_path = Path(X_test_path)
    y_pred_path = Path(y_pred_path)
    y_pred_prob_path = Path(y_pred_prob_path)

    # ---- 1. 载入模型和测试数据 ----
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)

    # ---- 2. 预测标签 -> 写入 y_pred_path ----
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame({"predicted_built_age": y_pred})
    y_pred_df.to_csv(y_pred_path, index=False)

    # ---- 3. 预测概率 -> 写入 y_pred_prob_path，并附带标签列 ----
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        prob_df = pd.DataFrame(
            y_prob,
            columns=[f"class_{i}" for i in range(y_prob.shape[1])]
        )
        # 🔴 关键：在概率表中也加上 predicted_built_age 列
        prob_df.insert(0, "predicted_built_age", y_pred)
        prob_df.to_csv(y_pred_prob_path, index=False)
    else:
        pd.DataFrame({"predicted_built_age": y_pred}).to_csv(
            y_pred_prob_path, index=False
        )


def pred_lda():
    base_dir = find_project_root()
    model_path = base_dir / "data_cache" / "models" / "lda_model.joblib"
    X_test_path = base_dir / "data_cache" / "energy_X_test.csv"
    y_pred_path = base_dir / "data_cache" / "models" / "lda_y_pred.csv"
    y_pred_prob_path = base_dir / "data_cache" / "models" / "lda_y_pred_prob.csv"
    predict(model_path, X_test_path, y_pred_path, y_pred_prob_path)


def pred_qda():
    base_dir = find_project_root()
    model_path = base_dir / "data_cache" / "models" / "qda_model.joblib"
    X_test_path = base_dir / "data_cache" / "energy_X_test.csv"
    y_pred_path = base_dir / "data_cache" / "models" / "qda_y_pred.csv"
    y_pred_prob_path = base_dir / "data_cache" / "models" / "qda_y_pred_prob.csv"
    predict(model_path, X_test_path, y_pred_path, y_pred_prob_path)
