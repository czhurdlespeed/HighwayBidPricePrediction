import numpy as np
import pandas as pd
from geopy.distance import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


def compute_similarity_score(embedding_a, embedding_b):
    return cosine_similarity(
        embedding_a.reshape(1, -1), embedding_b.reshape(1, -1)
    )[0][0]


def calculate_distance(lat1, long1, lat2, long2):
    return distance((lat1, long1), (lat2, long2)).miles


def percent_difference(value1, value2) -> float:
    if value1 == value2:
        return 0.0
    return ((value1 - value2) / ((value1 + value2) / 2)) * 100


def modify_aggregate_payitem_data(payitem_data):
    """
    For cases where a line item is listed multiple times on a test pay estimate, the data needs to be aggregated.
    """
    if len(payitem_data) > 1:
        total_quantity = payitem_data["Quantity"].sum()

        if "Unit Price" not in payitem_data.columns:
            aggregated_data = pd.DataFrame(
                {
                    "Item": [payitem_data["Item"].iloc[0]],
                    "Quantity": [total_quantity],
                }
            )
            for col in payitem_data.columns:
                if col not in ["Item", "Quantity"]:
                    aggregated_data[col] = payitem_data[col].iloc[0]
            return aggregated_data

        weighted_unit_price = np.average(
            payitem_data["Unit Price"], weights=payitem_data["Quantity"]
        )
        aggregated_data = pd.DataFrame(
            {
                "Item": [payitem_data["Item"].iloc[0]],
                "Quantity": [total_quantity],
                "Unit Price": [weighted_unit_price],
            }
        )
        # Take first value for columns with duplicate values (don't want to aggregate these (e.g. Description))
        for col in payitem_data.columns:
            if col not in ["Item", "Quantity", "Unit Price"]:
                aggregated_data[col] = payitem_data[col].iloc[0]

        return aggregated_data
    else:
        return payitem_data


def feature_vector_similarity(featurevector_a, featurevector_b):
    """
    Calculate similarity between two feature vectors
    """
    return cosine_similarity(
        featurevector_a.reshape(1, -1), featurevector_b.reshape(1, -1)
    ).item()


def similarity_to_confidence(similarity_score, method="sigmoid") -> float:
    if method == "linear":
        return min(100, max(0, similarity_score * 100))
    elif method == "sigmoid":
        import numpy as np

        return 100 / (1 + np.exp(-10 * (similarity_score - 0.5)))
    elif method == "threshold":
        if similarity_score > 0.8:
            return 90
        elif similarity_score > 0.6:
            return 70
        elif similarity_score > 0.4:
            return 50
        else:
            return 30
    else:
        raise ValueError(
            "Invalid method. Choose from 'linear', 'sigmoid', 'threshold'"
        )


def predict_with_confidence(knn_model, test_df, X_train, k=5):
    # Make prediction
    prediction = knn_model.predict(test_df)

    if len(X_train) < k:
        k = len(X_train)

    # Compute similarities
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(X_train)
    distances, indices = nn.kneighbors(test_df)
    similarities = 1 - distances

    # Compute confidence scores
    max_similarity = np.max(similarities, axis=1)
    avg_similarity = np.mean(similarities, axis=1)
    p95_similarity = np.percentile(similarities, 95, axis=1)

    # Convert to confidence percentages
    max_confidence = similarity_to_confidence(max_similarity, method="sigmoid")
    avg_confidence = similarity_to_confidence(avg_similarity, method="sigmoid")
    p95_confidence = similarity_to_confidence(p95_similarity, method="sigmoid")

    # Combine confidence scores (you can adjust the weights)
    combined_confidence = (
        max_confidence + 2 * avg_confidence + 2 * p95_confidence
    ) / 5

    return (
        prediction,
        combined_confidence,
        {
            "max_similarity": max_similarity,
            "avg_similarity": avg_similarity,
            "p95_similarity": p95_similarity,
            "max_confidence": max_confidence,
            "avg_confidence": avg_confidence,
            "p95_confidence": p95_confidence,
        },
    )


def interpret_confidence(confidence):
    if confidence > 95:
        return "Very High"
    elif confidence > 90:
        return "High"
    elif confidence > 80:
        return "Moderate"
    elif confidence > 70:
        return "Low"
    else:
        return "Very Low"
