from pathlib import Path
import joblib
import pandas as pd
from course.utils import find_project_root


def predict(model_path, X_test_path, y_pred_path, y_pred_prob_path):
    """
    Run a classifier on test data and save:

    - y_pred_path: CSV with a single column 'predicted_built_age' (class labels)
    - y_pred_prob_path: CSV with probability columns 'class_0', 'class_1', ...

    注意：
    * 如果 y_pred_prob_path 与 y_pred_path 相同，则只写标签文件，
      避免覆盖掉 'predicted_built_age'（单元测试就是这么设计的）。
    """

    # ---- 统一 Path 对象 ----
    model_path = Path(model_path)
    X_test_path = Path(X_test_path)
    y_pred_path = Path(y_pred_path)
    y_pred_prob_path = Path(y_pred_prob_path)

    # ---- 1. 加载模型和测试数据 ----
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)

    # ---- 2. 预测标签：只给 y_pred.csv 写 predicted_built_age ----
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame({"predicted_built_age": y_pred})
    y_pred_df.to_csv(y_pred_path, index=False)

    # 路径是否相同？（同路径就不要再写概率文件，避免覆盖标签）
    same_output_path = (y_pred_prob_path.resolve() == y_pred_path.resolve())

    # ---- 3. 预测概率：只在路径不同的时候写 y_pred_prob.csv ----
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        prob_df = pd.DataFrame(
            y_prob,
            columns=[f"class_{i}" for i in range(y_prob.shape[1])]
        )
        # 确保是数值类型，后面 roc_curve 里做减法不会出 'str' - 'str'
        prob_df = prob_df.astype(float)

        if not same_output_path:
            prob_df.to_csv(y_pred_prob_path, index=False)
    else:
        # 没有 predict_proba 的模型，仅当路径不同才写一个空表占位
        if not same_output_path:
            pd.DataFrame().to_csv(y_pred_prob_path, index=False)


def pred_lda():
    """
    Wrapper used by dodo_supervised.task_predict_lda

    读取 LDA 模型和测试集，生成:
      - data_cache/models/lda_y_pred.csv
      - data_cache/models/lda_y_pred_prob.csv
    """
    base_dir = find_project_root()
    model_path = base_dir / "data_cache" / "models" / "lda_model.joblib"
    X_test_path = base_dir / "data_cache" / "energy_X_test.csv"
    y_pred_path = base_dir / "data_cache" / "models" / "lda_y_pred.csv"
    y_pred_prob_path = base_dir / "data_cache" / "models" / "lda_y_pred_prob.csv"
    predict(model_path, X_test_path, y_pred_path, y_pred_prob_path)


def pred_qda():
    """
    Wrapper used by dodo_supervised.task_predict_qda

    读取 QDA 模型和测试集，生成:
      - data_cache/models/qda_y_pred.csv
      - data_cache/models/qda_y_pred_prob.csv
    """
    base_dir = find_project_root()
    model_path = base_dir / "data_cache" / "models" / "qda_model.joblib"
    X_test_path = base_dir / "data_cache" / "energy_X_test.csv"
    y_pred_path = base_dir / "data_cache" / "models" / "qda_y_pred.csv"
    y_pred_prob_path = base_dir / "data_cache" / "models" / "qda_y_pred_prob.csv"
    predict(model_path, X_test_path, y_pred_path, y_pred_prob_path)
