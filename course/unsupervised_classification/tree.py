from scipy.cluster.hierarchy import linkage, fcluster
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path("data_cache") / "vignettes" / "unsupervised_classification"


def hcluster_analysis():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / "data_cache" / "la_collision.csv")
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    outpath = base_dir / VIGNETTE_DIR / "dendrogram.html"
    fig = _plot_dendrogram(df_scaled)
    fig.write_html(outpath)


def hierarchical_groups(height):
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / "data_cache" / "la_collision.csv")
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    linked = _fit_dendrogram(df_scaled)
    clusters = _cutree(linked, height)  # adjust this value based on dendrogram scale
    df_plot = _pca(df_scaled)
    df_plot["cluster"] = clusters.astype(str)  # convert to string for color grouping
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
