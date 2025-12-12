import pandas as pd
from sklearn.metrics import classification_report
from course.utils import find_project_root


def metric_report(y_test_path, y_pred_path, report_path):
    """Create a pandas data frame called report which contains your
    classifier results and save it to CSV at report_path.
    """
    # 读取真实标签和预测标签（单列 CSV）
    y_test = pd.read_csv(y_test_path).iloc[:, 0]
    y_pred = pd.read_csv(y_pred_path).iloc[:, 0]

    # 使用 sklearn 生成 classification report 的字典
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # 转成 DataFrame，行是类别/指标
    report = pd.DataFrame(report_dict).transpose()

    # 保存到 CSV
    report.to_csv(report_path, index=True)


def metric_report_lda():
    base_dir = find_project_root()
    y_test_path = base_dir / 'data_cache' / 'energy_y_test.csv'
    y_pred_path = base_dir / 'data_cache' / 'models' / 'lda_y_pred.csv'
    report_path = base_dir / 'vignettes' / 'supervised_classification' / 'lda.csv'
    metric_report(y_test_path, y_pred_path, report_path)


def metric_report_qda():
    base_dir = find_project_root()
    y_test_path = base_dir / 'data_cache' / 'energy_y_test.csv'
    y_pred_path = base_dir / 'data_cache' / 'models' / 'qda_y_pred.csv'
    report_path = base_dir / 'vignettes' / 'supervised_classification' / 'qda.csv'
    metric_report(y_test_path, y_pred_path, report_path)
