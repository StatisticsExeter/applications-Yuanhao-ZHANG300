from scipy.cluster.hierarchy import linkage, fcluster
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
from course.utils import find_project_root
import plotly.graph_objects as go

VIGNETTE_DIR = Path("data_cache") / "vignettes" / "unsupervised_classification"


def hcluster_analysis(df=None):
    """
    进行分层聚类分析：
    - 如果 df 为 None：从 data_cache/la_collision.csv 读取数据，主要用于 ./run reports 管线
    - 如果传入 df：使用给定数据做分析，并把处理后的 df 返回（方便你手动调试）
    """
    created_inside = False

    if df is None:
        base_dir = find_project_root()
        df = pd.read_csv(base_dir / "data_cache" / "la_collision.csv")
        created_inside = True

    # ---------- 1. 只取数值列并标准化 ----------
    numeric_cols = df.select_dtypes(include="number").columns
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[numeric_cols]),
        columns=numeric_cols,
        index=df.index,
    )

    # ---------- 2. 层次聚类 ----------
    Z = linkage(df_scaled, method="ward")

    # 例如切成 4 类（这个数可以按需要调整）
    clusters = fcluster(Z, t=4, criterion="maxclust")
    df["cluster"] = clusters

    # ---------- 3. PCA 降维到 2 维用于可视化 ----------
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(df_scaled)
    pcs_df = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=df.index)
    pcs_df["cluster"] = df["cluster"]

    # ---------- 4. 画图并保存到 vignettes 目录 ----------
    VIGNETTE_DIR.mkdir(parents=True, exist_ok=True)

    # （1）树状图
    fig_dendro = ff.create_dendrogram(
        df_scaled.values,
        labels=df.index.astype(str),
    )
    fig_dendro.update_layout(
        title="Interactive Hierarchical Clustering Dendrogram"
    )
    fig_dendro.write_html(VIGNETTE_DIR / "hcluster_dendrogram.html")

    # （2）PCA 聚类散点图
    fig_scatter = px.scatter(
        pcs_df,
        x="PC1",
        y="PC2",
        color=pcs_df["cluster"].astype(str),
        title="PCA Scatter Plot Colored by Cluster Labels",
    )
    fig_scatter.write_html(VIGNETTE_DIR / "hcluster_scatter.html")

    # ---------- 5. 返回值：对 DoIt 返回 True；对手动调用返回 df ----------
    if created_inside:
        # 被 ./run reports 调用：只需要告诉 DoIt “任务成功”
        return True
    else:
        # 你在 notebook / Python 里手动传 df 调用时，可以拿到带 cluster 的 df
        return df


def hierarchical_groups(height):
    base_dir = find_project_root()

    # 读原始数据
    df = pd.read_csv(base_dir / "data_cache" / "la_collision.csv")

    # 只拿数值列做聚类 + PCA
    numeric_df = df.select_dtypes(include=["number"])

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(numeric_df)

    # 层次聚类 & 切树
    linked = _fit_dendrogram(df_scaled)
    clusters_df = _cutree(linked, height)          # DataFrame，有一列 'cluster'

    # 做 PCA 降维后画散点
    df_plot = _pca(df_scaled)
    df_plot["cluster"] = clusters_df["cluster"].astype(str)

    # 输出 HTML 到 vignette 目录
    VIGNETTE_DIR.mkdir(parents=True, exist_ok=True)
    outpath = base_dir / VIGNETTE_DIR / "hscatter.html"
    fig = _scatter_clusters(df_plot)
    fig.write_html(outpath)



def _fit_dendrogram(df):
    """Given a dataframe/array containing only suitable values
    return a scipy.cluster.hierarchy hierarchical clustering solution."""
    # 使用 Ward linkage（测试里也是这样构造“真值”）
    linked = linkage(df, method="ward")
    return linked


def _plot_dendrogram(df):
    """Given a dataframe/array df containing only suitable variables
    use plotly.figure_factory to plot a dendrogram of these data and
    return the resulting Figure.
    """
    fig = ff.create_dendrogram(df)
    fig.update_layout(title_text="Interactive Hierarchical Clustering Dendrogram")
    return fig


def _cutree(tree, height):
    """Given a scipy.cluster.hierarchy hierarchical clustering solution and a float `height`
    cut the tree at that height and return the solution (cluster group membership) as a
    pandas Series or 1D array of cluster labels.
    Tests then wrap this in a DataFrame, so这里直接返回一维 array/Series 即可。
    """
    clusters = fcluster(tree, t=height, criterion="distance")
    # 测试里期望的是 DataFrame：一列叫 'cluster'
    clusters_df = pd.DataFrame({"cluster": clusters})
    return clusters_df


def _pca(df):
    """Given a dataframe/array of only suitable variables
    return a dataframe of the first two PCA projections (z values)
    with columns 'PC1' and 'PC2'."""
    pca = PCA(n_components=2)
    comps = pca.fit_transform(df)
    df_pca = pd.DataFrame(comps, columns=["PC1", "PC2"])
    return df_pca


def _scatter_clusters(df):
    """Given a data frame containing columns 'PC1', 'PC2' and 'cluster'
    (the first two principal component projections and the cluster groups),
    return a plotly express scatterplot of PC1 versus PC2
    with colour denoting cluster membership.
    """
    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="cluster",
        # ★ 测试要求的标题：
        title="PCA Scatter Plot Colored by Cluster Labels",
    )
    return fig
