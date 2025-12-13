from pathlib import Path
from course.unsupervised_classification.eda import plot_scatter
from course.unsupervised_classification.tree import (
    hierarchical_groups,
    hcluster_analysis,
)
from course.unsupervised_classification.kmeans import kmeans


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
        models_path = Path("data_cache/vignettes/unsupervised_classification")
        models_path.mkdir(parents=True, exist_ok=True)
    return {
        "actions": [check_cache_results]
    }


def task_crash_summaries():
    """
    本地版本：不再从数据库刷新数据，只检查 la_collision.csv 是否存在。
    """
    def crash_summaries():
        csv_path = Path("data_cache/la_collision.csv")
        if not csv_path.exists():
            raise FileNotFoundError(
                "Expected cached data at 'data_cache/la_collision.csv' but it was not found. "
                "Please copy the provided coursework CSV into this location."
            )

    return {
        "actions": [crash_summaries],
        "targets": ["data_cache/la_collision.csv"],
    }


def task_eda():
    return {
        "actions": [plot_scatter],
        "file_dep": [
            "data_cache/la_collision.csv",
            "course/unsupervised_classification/eda.py",
        ],
        "targets": [
            "data_cache/vignettes/unsupervised_classification/scatterplot.html"
        ],
    }


def task_hcluster_analysis():
    return {
        "actions": [hcluster_analysis],
        "file_dep": [
            "data_cache/la_collision.csv",
            "course/unsupervised_classification/tree.py",
        ],
        "targets": [
            "data_cache/vignettes/unsupervised_classification/dendrogram.html"
        ],
    }


def task_hierarchical_groups():
    return {
        "actions": [lambda: hierarchical_groups(20)],
        "file_dep": [
            "data_cache/la_collision.csv",
            "course/unsupervised_classification/tree.py",
        ],
        "targets": [
            "data_cache/vignettes/unsupervised_classification/hscatter.html"
        ],
    }


def task_kmeans():
    return {
        "actions": [lambda: kmeans(4)],
        "file_dep": [
            "data_cache/la_collision.csv",
            "course/unsupervised_classification/kmeans.py",
        ],
        "targets": [
            "data_cache/vignettes/unsupervised_classification/kscatter.html",
            "data_cache/vignettes/unsupervised_classification/kcentroids1.html",
            "data_cache/vignettes/unsupervised_classification/kcentroids2.html",
        ],
    }
