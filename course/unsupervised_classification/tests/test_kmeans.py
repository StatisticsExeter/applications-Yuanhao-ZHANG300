import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from course.unsupervised_classification.kmeans import _kmeans


def test_kmeans_returns_model_with_correct_clusters():
    df = pd.DataFrame(np.random.rand(20, 4))
    model = _kmeans(df, 3)
    assert isinstance(model, KMeans)
    assert model.n_clusters == 3
    assert len(model.labels_) == df.shape[0]
