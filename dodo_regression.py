from pathlib import Path
from course.regression.eda import boxplot_age, boxplot_rooms
from course.regression.fit_model import fit_model
from course.regression.caterpillar_reffs import plot_caterpillar


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
        models_path = Path("data_cache/vignettes/regression")
        models_path.mkdir(parents=True, exist_ok=True)
    return {
        "actions": [check_cache_results]
    }


def task_energy_metrics_la():
    """
    本地版本：不再从数据库刷新数据，只检查 la_energy.csv 是否存在。
    """
    def energy_metrics_la():
        csv_path = Path("data_cache/la_energy.csv")
        if not csv_path.exists():
            raise FileNotFoundError(
                "Expected cached data at 'data_cache/la_energy.csv' but it was not found. "
                "Please copy the provided coursework CSV into this location."
            )

    return {
        "actions": [energy_metrics_la],
        "targets": ["data_cache/la_energy.csv"],
    }


def task_eda():
    return {
        "actions": [boxplot_age, boxplot_rooms],
        "file_dep": [
            "data_cache/la_energy.csv",
            "course/regression/eda.py",
        ],
        "targets": [
            "data_cache/vignettes/regression/boxplot_age.html",
            "data_cache/vignettes/regression/boxplot_rooms.html",
        ],
    }


def task_fit_model():
    return {
        "actions": [fit_model],
        "file_dep": [
            "data_cache/la_energy.csv",
            "course/regression/fit_model.py",
        ],
        "targets": [
            "data_cache/vignettes/regression/model_fit.txt",
            "data_cache/models/reffs.csv",
        ],
    }


def task_caterpillar_plot():
    return {
        "actions": [plot_caterpillar],
        "file_dep": [
            "data_cache/models/reffs.csv",
            "data_cache/vignettes/regression/model_fit.txt",  # 作为依赖使用
            "course/regression/caterpillar_reffs.py",
        ],
        "targets": [
            "data_cache/vignettes/regression/caterpillar.html",  # 只保留这个输出
        ],
    }