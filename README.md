# Clustering Demos

A small collection of self-contained Jupyter notebooks showing different clustering and anomaly detection techniques, across multiple data types (tabular, text, time series, images, and audio).

The goal is to be **hands-on and minimal**: each notebook focuses on one method or idea, with code you can read and adapt quickly.

---

## Contents

### Classical clustering

- [`kmeans_scratch.ipynb`](kmeans_scratch.ipynb)  
  Implement K-Means “from scratch” (no high-level helper APIs), visualize the iterations, and build intuition for how the algorithm actually works.

- [`gmm_clustering.ipynb`](gmm_clustering.ipynb)  
  Gaussian Mixture Model clustering with soft assignments, showing how mixtures differ from hard K-Means and how to interpret responsibilities / probabilities.

- [`hierarchical_clustering.ipynb`](hierarchical_clustering.ipynb)  
  Agglomerative clustering with dendrograms and cluster cuts. Good for understanding distance/linkage choices and how hierarchy emerges from the data.

- [`dbscan_pycaret.ipynb`](dbscan_pycaret.ipynb)  
  Density-based clustering (DBSCAN) using a higher-level ML toolkit. Shows how to discover arbitrarily-shaped clusters and automatically treat noise as outliers.

### Anomaly & outlier detection

- [`anomaly_pyod.ipynb`](anomaly_pyod.ipynb)  
  Anomaly detection using specialized outlier-detection tooling. Compares different detectors and visualizes which points are considered anomalies.

### Text & documents

- [`document_clustering_llm.ipynb`](document_clustering_llm.ipynb)  
  Cluster text documents using modern embeddings (e.g. from LLM / sentence embedding models). Shows how to go from raw text → embeddings → clusters → topic inspection.

### Time series

- [`timeseries_clustering_embeddings.ipynb`](timeseries_clustering_embeddings.ipynb)  
  Cluster time series by embedding them into a vector space (for example via feature extraction or sequence encoders) and applying standard clustering on top.

### Images & audio (multimodal)

- [`image_clustering_imagebind.ipynb`](image_clustering_imagebind.ipynb)  
  Cluster images based on learned feature embeddings from a multimodal model (e.g. an ImageBind-style backbone), so visually/semantically similar images end up together.

- [`audio_clustering_imagebind.ipynb`](audio_clustering_imagebind.ipynb)  
  Similar idea as the image notebook, but for audio clips. Turn audio into embeddings, then use clustering to find similar sounds / categories without labels.

---

## Running the notebooks

From the repository directory:

```bash
jupyter lab
# or
jupyter notebook
```

Then open any of the `.ipynb` files listed above and execute the cells top-to-bottom.

If you’re using VS Code, you can just open the folder, click on a notebook, and run it with the built-in Jupyter support.

---
