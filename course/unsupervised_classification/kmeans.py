import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from course.utils import find_project_root
from course.unsupervised_classification.tree import _scatter_clusters, _pca

VIGNETTE_DIR = Path("data_cache") / "vignettes" / "unsupervised_classification"


def _kmeans(df, k: int) -> KMeans:
    """Given dataframe/array `df` containing only suitable variables and
    integer `k`, return a scikit-learn KMeans solution fitted to these data.
    """
    # 测试里会传入 DataFrame(np.random.rand(...))，这里直接让 KMeans 自己做转换
    model = KMeans(n_clusters=k, random_state=0, n_init=10)
    model.fit(df)
    return model


def kmeans(n_clusters: int = 4):
    """
    课程流水线用的函数：
    1. 读取 data_cache/energy.csv
    2. 只选数值特征做标准化和聚类
    3. 把聚类标签写回原始 df 的 'cluster' 列
    4. 存成 data_cache/energy_kmeans.csv
    """
    data_path = Path("data_cache/energy.csv")
    df = pd.read_csv(data_path)

    # ---- 1. 选取特征列：去掉 ID 和已有的 cluster 列，只保留数值列 ----
    feature_df = df.drop(columns=["lad_cd", "cluster"], errors="ignore")
    feature_df = feature_df.select_dtypes(include="number")

    # ---- 2. 调用已经通过测试的 _kmeans 做聚类 ----
    model = _kmeans(feature_df, n_clusters)

    # ---- 3. 把标签写回原始 df ----
    df["cluster"] = model.labels_

    # ---- 4. 写出带聚类结果的缓存文件，供后续 tree / 报告使用 ----
    out_path = Path("data_cache/energy_kmeans.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # doit 要求 Python action 返回 True / None / 字符串 / dict
    return True


def _plot_centroids(scaled_centers, scaler, colnames, k):  # Melt for grouped bar plot
    original_centers = scaler.inverse_transform(scaled_centers)
    centers_df = pd.DataFrame(original_centers, columns=colnames).iloc[:, [0]]
    centers_df["cluster"] = [f"Cluster {i}" for i in range(k)]
    centers_melted = centers_df.melt(
        id_vars="cluster", var_name="Feature", value_name="Value"
    )
    fig1 = px.bar(
        centers_melted,
        x="Feature",
        y="Value",
        color="cluster",
        barmode="group",
        title="Cluster Centers by Feature (Original Scale)",
    )
    centers_df = pd.DataFrame(original_centers, columns=colnames).iloc[:, 1:]
    centers_df["cluster"] = [f"Cluster {i}" for i in range(k)]
    centers_melted = centers_df.melt(
        id_vars="cluster", var_name="Feature", value_name="Value"
    )
    fig2 = px.bar(
        centers_melted,
        x="Feature",
        y="Value",
        color="cluster",
        barmode="group",
        title="Cluster Centers by Feature (Original Scale)",
    )
    return fig1, fig2
