from pathlib import Path
from course.supervised_classification.classify import fit_qda, fit_lda
from course.supervised_classification.predict import pred_lda, pred_qda
from course.supervised_classification.metrics import metric_report_lda, metric_report_qda
from course.supervised_classification.split_test_train import test_and_train
from course.supervised_classification.eda import plot_scatter, get_summary_stats
from course.supervised_classification.roc_curve import plot_roc_curve


def task_check_cache_data():
    def check_cache_data():
        """Check cache folder exists"""
        models_path = Path("data_cache/models")
        models_path.mkdir(parents=True, exist_ok=True)
    return {
        "actions": [check_cache_data]
    }


def task_check_cache_results():
    def check_cache_results():
        """Check cache folder exists"""
        models_path = Path("data_cache/vignettes/supervised_classification")
        models_path.mkdir(parents=True, exist_ok=True)
    return {
        "actions": [check_cache_results]
    }


def task_energy_metrics_new():
    """
    本地版本：不再从数据库刷新数据，只检查缓存 CSV 是否存在。

    如果课程仓库中已经提供 data_cache/energy.csv，
    这里什么都不改写，只是确保文件存在；若不存在就给出清晰报错。
    """
    def energy_metrics_new():
        csv_path = Path("data_cache/energy.csv")
        if not csv_path.exists():
            raise FileNotFoundError(
                "Expected cached data at 'data_cache/energy.csv' but it was not found. "
                "Please copy the provided coursework CSV into this location."
            )

    return {
        "actions": [energy_metrics_new],
        "targets": ["data_cache/energy.csv"],
    }


def task_eda():
    return {
        "actions": [plot_scatter],
        "file_dep": ["data_cache/energy.csv"],
        "targets": ["data_cache/vignettes/supervised_classification/scatterplot.html"],
    }


def task_tabulate_stats():
    return {
        "actions": [get_summary_stats],
        "file_dep": [
            "data_cache/energy.csv",
            "course/supervised_classification/eda.py",
        ],
        "targets": [
            "data_cache/vignettes/supervised_classification/frequencies.csv",
            "data_cache/vignettes/supervised_classification/grouped_stats.csv",
        ],
    }


def task_test_and_train():
    return {
        "actions": [test_and_train],
        "file_dep": [
            "data_cache/energy.csv",
            "course/supervised_classification/split_test_train.py",
        ],
        "targets": [
            "data_cache/energy_X_train.csv",
            "data_cache/energy_y_train.csv",
            "data_cache/energy_X_test.csv",
            "data_cache/energy_y_test.csv",
        ],
    }


def task_fit_lda():
    return {
        "actions": [fit_lda],
        "file_dep": [
            "data_cache/energy_X_train.csv",
            "data_cache/energy_y_train.csv",
            "course/supervised_classification/classify.py",
        ],
        "targets": ["data_cache/models/lda_model.joblib"],
    }


def task_fit_qda():
    return {
        "actions": [fit_qda],
        "file_dep": [
            "data_cache/energy_X_train.csv",
            "data_cache/energy_y_train.csv",
            "course/supervised_classification/classify.py",
        ],
        "targets": ["data_cache/models/qda_model.joblib"],
    }


def task_predict_lda():
    return {
        "actions": [pred_lda],
        "file_dep": [
            "data_cache/models/lda_model.joblib",
            "data_cache/energy_X_test.csv",
            "course/supervised_classification/predict.py",
        ],
        "targets": [
            "data_cache/models/lda_y_pred.csv",
            "data_cache/models/lda_y_pred_prob.csv",
        ],
    }


def task_predict_qda():
    return {
        "actions": [pred_qda],
        "file_dep": [
            "data_cache/models/qda_model.joblib",
            "data_cache/energy_X_test.csv",
            "course/supervised_classification/predict.py",
        ],
        "targets": [
            "data_cache/models/qda_y_pred.csv",
            "data_cache/models/qda_y_pred_prob.csv",
        ],
    }


def task_metrics_lda():
    return {
        "actions": [metric_report_lda],
        "file_dep": [
            "data_cache/models/lda_y_pred.csv",
            "data_cache/energy_y_test.csv",
            "course/supervised_classification/metrics.py",
        ],
        "targets": ["data_cache/vignettes/supervised_classification/lda.csv"],
    }


def task_metrics_qda():
    return {
        "actions": [metric_report_qda],
        "file_dep": [
            "data_cache/models/qda_y_pred.csv",
            "data_cache/energy_y_test.csv",
            "course/supervised_classification/metrics.py",
        ],
        "targets": ["data_cache/vignettes/supervised_classification/qda.csv"],
    }


def task_roc_curve():
    return {
        "actions": [plot_roc_curve],
        "file_dep": [
            "data_cache/energy_y_test.csv",
            "data_cache/models/qda_y_pred_prob.csv",
        ],
        "targets": ["data_cache/vignettes/supervised_classification/roc.html"],
    }
