import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


def find_optimal_clusters(embeddings, max_clusters=20):
    silhouette_scores = []
    n_clusters_range = range(2, max_clusters + 1)

    for n_clusters in n_clusters_range:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="euclidean",  # changed from precomputed
            linkage="average",
        )
        cluster_labels = clustering.fit_predict(embeddings)
        # Calculate silhouette score using cosine metric directly
        silhouette_avg = silhouette_score(
            embeddings, cluster_labels, metric="cosine"
        )
        silhouette_scores.append(silhouette_avg)
        print(f"Silhouette score for {n_clusters} clusters: {silhouette_avg}")

    optimal_clusters = n_clusters_range[np.argmax(silhouette_scores)]
    return optimal_clusters


def plot_embeddings(filepath: Path, font_size=6, max_clusters=10):
    plot_path = Path("plots")
    plot_path.mkdir(exist_ok=True)

    # Load data
    with open(filepath, "r") as f:
        data = json.load(f)

    labels = [instance["index"] for instance in data["data"]]
    embeddings = [instance["embedding"] for instance in data["data"]]
    embeddings = np.array(embeddings)

    # Find optimal number of clusters
    n_clusters = find_optimal_clusters(embeddings, max_clusters)
    print(f"\nOptimal number of clusters: {n_clusters}")

    # Perform clustering with optimal number
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="euclidean",  # changed from precomputed
        linkage="average",
    )
    clusters = clustering.fit_predict(embeddings)

    # Create and fit TSNE
    tsne = TSNE(
        n_components=2,
        random_state=0,
        perplexity=min(30, len(embeddings) - 1),
        metric="cosine",
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot with improved styling
    plt.figure(figsize=(12, 12))

    # Create scatter plot with clusters
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=clusters,
        cmap="viridis",
        alpha=0.6,
    )

    # Add labels with smaller font
    for i, label in enumerate(labels):
        plt.annotate(
            label,
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=font_size,
            alpha=0.7,
        )

    # Add colorbar
    plt.colorbar(scatter, label=f"Cluster (k={n_clusters})")

    plt.title(
        f"t-SNE visualization of embeddings with {n_clusters} clusters\n(using cosine similarity)"
    )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(plot_path / "tsne_clusters.png", dpi=300, bbox_inches="tight")

    # Plot silhouette scores
    plt.figure(figsize=(10, 5))
    silhouette_scores = []
    n_clusters_range = range(2, max_clusters + 1)

    for k in n_clusters_range:
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric="euclidean",  # changed from precomputed
            linkage="average",
        )
        cluster_labels = clustering.fit_predict(embeddings)
        silhouette_scores.append(
            silhouette_score(embeddings, cluster_labels, metric="cosine")
        )

    plt.plot(n_clusters_range, silhouette_scores, "bo-")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title(
        "Silhouette Score vs Number of Clusters\n(using cosine similarity)"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        plot_path / "silhouette_scores.png", dpi=300, bbox_inches="tight"
    )


def find_closest_neighbors(
    filepath: Path, target_label: str, n_neighbors: int = 5
):
    # Load embeddings
    with open(filepath, "r") as f:
        data = json.load(f)

    # Create dictionary of label -> embedding
    embeddings_dict = {
        instance["index"]: instance["embedding"] for instance in data["data"]
    }

    if target_label not in embeddings_dict:
        raise ValueError(f"Label '{target_label}' not found in embeddings")

    # Get target embedding and all other embeddings
    target_embedding = np.array(embeddings_dict[target_label]).reshape(1, -1)

    # Calculate similarities
    similarities = []
    for label, embedding in embeddings_dict.items():
        if label != target_label:  # Skip comparing with itself
            similarity = cosine_similarity(
                target_embedding, np.array(embedding).reshape(1, -1)
            )[0][0]
            similarities.append((label, similarity))

    # Sort by similarity and get top n_neighbors
    closest = sorted(similarities, key=lambda x: x[1], reverse=True)[
        :n_neighbors
    ]

    # Print results nicely formatted
    print(f"\nTop {n_neighbors} closest neighbors to '{target_label}':")
    print("-" * 50)
    for label, similarity in closest:
        print(f"Label: {label:<30} Similarity: {similarity:.4f}")

    return closest


if __name__ == "__main__":
    filepath = Path("embeddings") / "jina_embedding_proposalIndex_response.json"
    plot_embeddings(filepath, font_size=6, max_clusters=20)
    subprocess.run(["open", "plots/tsne_clusters.png"])
    subprocess.run(["open", "plots/silhouette_scores.png"])

    input_label = input("Enter the target label: ")
    data_path = Path("nc_csv")
    while input_label != "exit":
        neighbors = find_closest_neighbors(filepath, input_label)
        open_excels = input("Open Excel files? (y/n): ")
        if open_excels == "y":
            subprocess.run(["open", data_path / f"{input_label}.csv"])
            for label, _ in neighbors[:3]:
                subprocess.run(["open", data_path / f"{label}.csv"])
        input_label = input("Enter the target label: ")
