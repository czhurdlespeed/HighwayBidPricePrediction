import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors


class ImprovedKNNRegression:
    def __init__(self, X, y, max_k=None, cv_repeats=5):
        self.X = X.copy()
        self.y = y.copy()
        self.feature_names = list(X.columns)
        self.n_samples = len(X)
        self.max_k = max_k or min(5, self.n_samples // 2)
        self.cv_repeats = cv_repeats
        self.best_k = self.find_best_k()
        self.model = self.train()

    def custom_cv_score(self, k):
        rkf = RepeatedKFold(
            n_splits=5, n_repeats=self.cv_repeats, random_state=42
        )
        mse_scores = []
        for train_index, test_index in rkf.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            n_neighbors = min(k, len(X_train) - 1)
            knn = KNeighborsRegressor(
                n_neighbors=n_neighbors, weights="distance", p=2
            )
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            mse_scores.append(mean_squared_error(y_test, y_pred))
        return np.mean(mse_scores)

    def find_best_k(self):
        if self.n_samples <= 5:
            return 1
        k_range = range(1, self.max_k + 1)
        mse_scores = [self.custom_cv_score(k) for k in k_range]
        best_k = k_range[np.argmin(mse_scores)]

        # Adaptive k selection
        density_based_k = self.adaptive_k_selection()

        # Ensemble approach
        ensemble_k = self.ensemble_k_selection(k_range, mse_scores)

        # Choose the final k (using weighted average)
        candidate_ks = [best_k, density_based_k, ensemble_k]
        weights = [0.5, 0.25, 0.25]  # Giving more weight to cross-validation
        final_k = int(np.average(candidate_ks, weights=weights))

        return max(1, min(final_k, self.n_samples - 1))

    def adaptive_k_selection(self):
        nn = NearestNeighbors(n_neighbors=min(20, self.n_samples - 1))
        nn.fit(self.X)
        distances, _ = nn.kneighbors(self.X)
        mean_distances = distances.mean(axis=1)
        k = int(np.ceil(1 / mean_distances.mean()))
        return min(k, self.max_k)

    def ensemble_k_selection(self, k_range, mse_scores):
        top_k_indices = np.argsort(mse_scores)[:3]
        return int(np.mean([k_range[i] for i in top_k_indices]))

    def train(self):
        k = self.best_k
        knn = KNeighborsRegressor(
            n_neighbors=k, weights="distance", p=2, n_jobs=-1
        )
        knn.fit(self.X, self.y)
        return knn

    def predict(self, X_test):
        X_test = X_test.reindex(columns=self.feature_names)
        return self.model.predict(X_test)
