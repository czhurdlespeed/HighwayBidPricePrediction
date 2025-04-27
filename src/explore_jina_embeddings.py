import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


def find_optimal_parameters(embeddings):
	silhouette_scores = []
	valid_combinations = []

	eps_range = np.arange(0.1, 5.1, 0.1)
	min_samples_range = np.arange(2, 20, 1)
	for eps in eps_range:
		for min_samples in min_samples_range:
			clustering = DBSCAN(eps=eps, min_samples=min_samples)
			cluster_labels = clustering.fit_predict(embeddings)
			
			# Skip if there's only one cluster or all points are noise
			n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
			if n_clusters < 2:
				print(f"Skipping eps={eps}, min_samples={min_samples} - only {n_clusters} clusters")
				silhouette_scores.append(-1)  # Use -1 to indicate invalid combination
				continue
				
			# Calculate silhouette score using cosine metric directly
			try:
				silhouette_avg = silhouette_score(
					embeddings, cluster_labels, metric="cosine"
				)
				silhouette_scores.append(silhouette_avg)
				valid_combinations.append((eps, min_samples))
				print(
					f"Silhouette score for eps={eps} and min_samples={min_samples}: {silhouette_avg}"
				)
			except ValueError:
				print(f"Error calculating silhouette score for eps={eps}, min_samples={min_samples}")
				silhouette_scores.append(-1)

	# Find the best valid combination
	valid_scores = [score for score in silhouette_scores if score != -1]
	if not valid_scores:
		print("Warning: No valid clustering found. Using default parameters.")
		return 0.5, 5  # Default values
		
	best_idx = np.argmax(valid_scores)
	return valid_combinations[best_idx]


def plot_embeddings(filepath: Path):
	plot_path = Path("plots")
	plot_path.mkdir(exist_ok=True)

	# Load data
	with open(filepath, "r") as f:
		data = json.load(f)

	labels = [instance["index"] for instance in data["data"]]
	embeddings = [instance["embedding"] for instance in data["data"]]
	embeddings = np.array(embeddings)

	# Find optimal parameters
	eps, min_samples = find_optimal_parameters(embeddings)
	print(f"\nOptimal parameters: eps={eps}, min_samples={min_samples}")

	# Perform clustering with optimal parameters
	clustering = DBSCAN(eps=eps, min_samples=min_samples)
	clusters = clustering.fit_predict(embeddings)

	# Create and fit TSNE
	tsne = TSNE(
		n_components=2,
		random_state=0,
		metric="cosine",
	)
	embeddings_2d = tsne.fit_transform(embeddings)

	plt.figure(figsize=(12, 12), layout="constrained")

	# Create scatter plot with clusters
	# First plot noise points in black
	noise_mask = clusters == -1
	if np.any(noise_mask):
		plt.scatter(
			embeddings_2d[noise_mask, 0],
			embeddings_2d[noise_mask, 1],
			c='red',
			alpha=0.6,
			label='Noise'
		)
	
	# Then plot clustered points with colors
	cluster_mask = ~noise_mask
	if np.any(cluster_mask):
		scatter = plt.scatter(
			embeddings_2d[cluster_mask, 0],
			embeddings_2d[cluster_mask, 1],
			c=clusters[cluster_mask],
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
			fontsize="xx-small",
			alpha=0.7,
		)

	# Create legend
	unique_clusters = np.unique(clusters[cluster_mask]) if np.any(cluster_mask) else []
	legend_elements = []
	
	# Add noise to legend if present
	if np.any(noise_mask):
		legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
										markerfacecolor='red', 
										label='Noise', markersize=10))
	
	# Add clusters to legend
	for cluster in unique_clusters:
		legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
										markerfacecolor=plt.cm.viridis(cluster/len(unique_clusters)), 
										label=f'Cluster {cluster}', markersize=10))
	
	plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

	plt.title(
		f"t-SNE visualization of embeddings with {len(np.unique(clusters))} clusters\n(using cosine similarity)"
	)
	plt.xticks([])
	plt.yticks([])
	plt.savefig(plot_path / "tsne_clusters.png", dpi=300, bbox_inches="tight")

	# Plot silhouette scores
	plt.figure(figsize=(10, 5), layout="constrained")
	silhouette_scores = []
	valid_combinations = []

	eps_range = np.arange(0.1, 5.1, 0.1)
	min_samples_range = np.arange(2, 20, 1)
	for eps in eps_range:
		for min_samples in min_samples_range:
			clustering = DBSCAN(eps=eps, min_samples=min_samples)
			cluster_labels = clustering.fit_predict(embeddings)
			
			# Skip if there's only one cluster or all points are noise
			n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
			if n_clusters < 2:
				silhouette_scores.append(-1)  # Use -1 to indicate invalid combination
				continue
				
			try:
				score = silhouette_score(embeddings, cluster_labels, metric="cosine")
				silhouette_scores.append(score)
				valid_combinations.append((eps, min_samples))
			except ValueError:
				silhouette_scores.append(-1)

	# Reshape silhouette scores into a 2D array for the heatmap
	scores_matrix = np.array(silhouette_scores).reshape(len(eps_range), len(min_samples_range))
	
	# Create heatmap
	plt.figure(figsize=(14, 10)) 
	im = plt.imshow(scores_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
	
	# Add colorbar
	plt.colorbar(im, label='Silhouette Score')
	
	# Set ticks and labels
	plt.xticks(np.arange(len(min_samples_range)), min_samples_range)
	plt.yticks(np.arange(len(eps_range)), np.round(eps_range, 1))
	plt.xlabel('Min Samples')
	plt.ylabel('Epsilon (eps)')
	plt.title('Silhouette Scores Heatmap for DBSCAN Parameters')
	
	# Add text annotations
	for i in range(len(eps_range)):
		for j in range(len(min_samples_range)):
			score = scores_matrix[i, j]
			if score != -1:  # Only add text for valid combinations
				plt.text(j, i, f'{score:.2f}',
						ha='center', va='center', color='white' if score < 0.5 else 'black')
	
	plt.tight_layout()
	plt.savefig(plot_path / "silhouette_heatmap.png", dpi=300, bbox_inches="tight")
	plt.close()



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
	plot_embeddings(filepath)
	subprocess.run(["open", "plots/tsne_clusters.png"])
	subprocess.run(["open", "plots/silhouette_heatmap.png"])
	exit()

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
