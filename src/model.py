import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model_helpers import (
    PayItemBid,
    combine_csv_files,
    load_distance,
    loop_contractor_payitems_inference_contract,
    return_historic_bids_for_payitem,
)


def convert_date_to_timestamp(date_str: str) -> float:
    return datetime.strptime(date_str, "%m/%d/%Y").timestamp() / (24 * 3600)


class WeightedKNNRegression(nn.Module):
    def __init__(self, k: int = 5, learning_rate: float = 0.01):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.k = k

    def compute_similarity_matrix(
        self,
        query_features: Dict[str, torch.Tensor],
        reference_features: Dict[str, torch.Tensor],
        is_inference: bool = False,
    ) -> torch.Tensor:
        """
        embedding_sim = -torch.cdist(
            query_features["embeddings"].unsqueeze(0),
            reference_features["embeddings"],
        ).squeeze(0)
        """

        distance_sim = -torch.cdist(
            query_features["distances"].view(1, 1),
            reference_features["distances"],
        ).squeeze(0)

        quantity_sim = -torch.cdist(
            query_features["quantities"].view(1, 1),
            reference_features["quantities"],
        ).squeeze(0)

        bid_size_sim = -torch.cdist(
            query_features["bid_size"].view(1, 1),
            reference_features["bid_size"],
        ).squeeze(0)

        similarities = torch.stack(
            [
                self.weights[0] * distance_sim,
                self.weights[1] * quantity_sim,
                self.weights[2] * bid_size_sim,
            ]
        )

        final_similarity = torch.softmax(similarities, dim=0)
        return torch.mean(final_similarity, dim=0)

    def predict_price(
        self, similarities: torch.Tensor, prices: torch.Tensor
    ) -> torch.Tensor:
        top_k_sims, indices = torch.topk(
            similarities, k=min(self.k, similarities.size(0))
        )
        prices = prices.view(-1)
        top_k_prices = prices[indices]
        weights = 1 / (1 - top_k_sims + 1e-6)
        weights = weights / weights.sum()
        return torch.dot(top_k_prices, weights)

    def forward(
        self,
        query_features: Dict[str, torch.Tensor],
        reference_features: Dict[str, torch.Tensor],
        is_inference: bool = False,
    ) -> torch.Tensor:
        similarities = self.compute_similarity_matrix(
            query_features, reference_features, is_inference
        )
        predicted_prices = self.predict_price(
            similarities, reference_features["prices"]
        )
        return predicted_prices

    def train_step(self, query_features, reference_features, target_prices):
        self.optimizer.zero_grad()
        predicted_prices = self(query_features, reference_features)
        loss = nn.HuberLoss()(predicted_prices.view(-1), target_prices.view(-1))
        loss.backward()
        # self.optimizer.step()
        return loss.item()


class PayItemPredictor:
    def __init__(self, pay_item_id: str):
        self.pay_item_id = pay_item_id
        self.model = None
        self.zero_tensor = torch.tensor([0.0], dtype=torch.float32)
        self.false_tensor = torch.tensor([False], dtype=torch.bool)

    def filter_reference_data(
        self,
        query_bid: PayItemBid,
        reference_bids: List[PayItemBid],
    ) -> List[PayItemBid]:
        query_timestamp = convert_date_to_timestamp(query_bid.bid_date)

        filtered_bids = []
        for bid in reference_bids:
            if bid.contract == query_bid.contract:
                continue
            bid_timestamp = convert_date_to_timestamp(bid.bid_date)
            time_diff = abs(bid_timestamp - query_timestamp)
            query_tensor = torch.tensor(
                query_bid.embedding, dtype=torch.float32
            )
            bid_tensor = torch.tensor(bid.embedding, dtype=torch.float32)
            cos_sim = torch.dot(query_tensor, bid_tensor) / (
                torch.norm(query_tensor) * torch.norm(bid_tensor)
            )
            if (
                bid.contractor_id == query_bid.contractor_id
                or (bid.is_winning and time_diff <= 120)
                or (
                    bid.is_winning
                    and bid.contractor_id in query_bid.competitors
                )
                or (
                    (
                        query_bid.quantity < 1.10 * bid.quantity
                        and query_bid.quantity > 0.90 * bid.quantity
                    )
                )
                and (cos_sim > 0.80)
            ):
                filtered_bids.append(bid)

        if not filtered_bids:
            raise ValueError(
                f"No suitable reference bids found for contractor {query_bid.contractor_id}"
            )

        return filtered_bids

    def prepare_features(
        self,
        bid: PayItemBid,
        filtered_bids: List[PayItemBid] = None,
        is_inference: bool = False,
        query_contract: str = None,
        is_query: bool = True,
    ) -> Dict[str, torch.Tensor]:
        distance = self.zero_tensor.clone()

        if query_contract and bid.contract != query_contract:
            distance = torch.tensor(
                [load_distance(query_contract, bid.contract)],
                dtype=torch.float32,
            )

        price = 0.0 if (is_inference and is_query) else bid.unit_price

        return {
            "embeddings": torch.tensor(bid.embedding, dtype=torch.float32),
            "distances": distance,
            "quantities": torch.tensor([bid.quantity], dtype=torch.float32),
            "prices": torch.tensor([price], dtype=torch.float32),
            "bid_size": torch.tensor([bid.bid_size], dtype=torch.float32),
            "is_winning": torch.tensor(
                [bid.is_winning if not is_inference else False],
                dtype=torch.bool,
            ),
        }

    def prepare_batch(
        self,
        bids: List[PayItemBid],
        filtered_bids: List[PayItemBid] = None,
        is_inference: bool = False,
        query_contract: str = None,
    ) -> Dict[str, torch.Tensor]:
        features = {
            "embeddings": [],
            "distances": [],
            "quantities": [],
            "bid_size": [],
            "prices": [],
            "is_winning": [],
        }

        for bid in bids:
            bid_features = self.prepare_features(
                bid,
                filtered_bids,
                is_inference,
                query_contract,
                is_query=False,
            )
            for key in features:
                features[key].append(bid_features[key])

        batched = {}
        for key in features:
            if key == "embeddings":
                batched[key] = torch.stack(features[key])
            else:
                batched[key] = torch.stack(features[key]).view(-1, 1)

        return batched

    @staticmethod
    def contract_aware_split(bids, validation_split=0.2, random_state=None):
        """
        Split bids into training and validation sets ensuring no contract appears in both sets.

        Args:
            bids: List of bid dictionaries, each containing 'contract_id'
            validation_split: Float between 0 and 1, fraction of contracts for validation
            random_state: Optional random seed for reproducibility

        Returns:
            train_bids: List of bids for training
            val_bids: List of bids for validation
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Group bids by contract_id
        contract_to_bids = {}
        for bid in bids:
            contract_id = bid.contract
            if contract_id not in contract_to_bids:
                contract_to_bids[contract_id] = []
            contract_to_bids[contract_id].append(bid)

        # Get unique contract IDs and shuffle them
        contract_ids = list(contract_to_bids.keys())
        np.random.shuffle(contract_ids)

        # Split contracts
        split_idx = int(len(contract_ids) * (1 - validation_split))
        train_contract_ids = set(contract_ids[:split_idx])

        # Assign bids based on their contract ID
        train_bids = []
        val_bids = []

        for bid in bids:
            if bid.contract in train_contract_ids:
                train_bids.append(bid)
            else:
                val_bids.append(bid)

        return train_bids, val_bids

    def train(
        self,
        training_bids: List[PayItemBid],
        validation_split: float = 0.1,
        n_epochs: int = 100,
        k_range: List[int] = None,
    ):
        train_bids, val_bids = self.contract_aware_split(
            training_bids, validation_split=validation_split
        )
        if k_range is None:
            max_k = min(10, len(train_bids) - 1)
            k_range = list(range(1, max_k + 1))

        k_results = {}
        best_overall_k = None
        best_overall_loss = float("inf")

        for k in k_range:
            print(f"\nTraining model with k={k}")
            self.model = WeightedKNNRegression(k=k)
            best_val_loss = float("inf")
            best_train_loss = float("inf")
            best_weights = None

            for epoch in range(n_epochs):
                self.model.optimizer.zero_grad()
                epoch_loss = 0
                n_samples = 0

                for query_bid in train_bids:
                    try:
                        filtered_bids = self.filter_reference_data(
                            query_bid, train_bids
                        )
                        # print(f"\nTraining set size: {len(filtered_bids)}")
                        query_features = self.prepare_features(
                            query_bid, filtered_bids
                        )
                        reference_features = self.prepare_batch(
                            filtered_bids,
                            filtered_bids,
                            query_contract=query_bid.contract,
                        )
                        loss = self.model.train_step(
                            query_features,
                            reference_features,
                            query_features["prices"],
                        )
                        epoch_loss += loss
                        n_samples += 1
                    except ValueError:
                        continue
                self.model.optimizer.step()

                val_loss = 0
                n_val_samples = 0
                for val_bid in val_bids:
                    try:
                        filtered_bids = self.filter_reference_data(
                            val_bid, train_bids
                        )
                        # print(f"Validation set size: {len(filtered_bids)}")
                        val_features = self.prepare_features(
                            val_bid, filtered_bids
                        )
                        reference_features = self.prepare_batch(
                            filtered_bids,
                            filtered_bids,
                            query_contract=val_bid.contract,
                        )
                        with torch.no_grad():
                            predictions = self.model(
                                val_features, reference_features
                            )
                            print(f"Predicted price: {predictions.item()}")
                            print(
                                f"Actual price: {val_features['prices'].item()}"
                            )
                            val_loss += nn.HuberLoss()(
                                predictions.view(-1),
                                val_features["prices"].view(-1),
                            ).item()
                            n_val_samples += 1
                    except ValueError:
                        continue

                avg_train_loss = (
                    epoch_loss / n_samples if n_samples > 0 else float("inf")
                )
                avg_val_loss = (
                    val_loss / n_val_samples
                    if n_val_samples > 0
                    else float("inf")
                )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_weights = self.model.state_dict()

                if avg_train_loss < best_train_loss:
                    best_train_loss = avg_train_loss

                if epoch % 1 == 0:
                    print(
                        f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
                    )
                print(f"Best train loss: {best_train_loss:.4f}")
                print(f"Best val loss: {best_val_loss:.4f}")
                # Print individual feature weights
                print_weights = (
                    torch.softmax(self.model.weights, dim=0).detach().numpy()
                )

                class FeatureWeights(Enum):
                    Distance = 0
                    Quantity = 1
                    BidSize = 2

                for i, weight in enumerate(print_weights):
                    print(f"{FeatureWeights(i).name}: {weight:.4f}")
            k_results[k] = {
                "val_loss": best_val_loss,
                "weights": best_weights,
                "feature_weights": torch.softmax(self.model.weights, dim=0)
                .detach()
                .numpy(),
            }

            if best_val_loss < best_overall_loss:
                best_overall_loss = best_val_loss
                best_overall_k = k
                print(f"Saving model for k={k}")
                shutil.rmtree("models")
                Path("models").mkdir(exist_ok=True)
                torch.save(
                    best_weights, f"models/{train_bids[0].payitem}_{k}.pt"
                )
        print("\nResults for all k values:")
        for k, results in k_results.items():
            print(f"k={k}: Validation Loss = {results['val_loss']:.4f}")
            print(f"Feature weights: {results['feature_weights']}")

        print(f"\nBest k value: {best_overall_k}")
        print(f"Best Validation loss: {best_overall_loss:.4f}")
        self.model = WeightedKNNRegression(k=best_overall_k)
        self.model.load_state_dict(k_results[best_overall_k]["weights"])

        return k_results

    def predict(
        self, query_bid: PayItemBid, reference_bids: List[PayItemBid]
    ) -> float:

        filtered_bids = self.filter_reference_data(query_bid, reference_bids)

        query_features = self.prepare_features(
            query_bid, filtered_bids, is_inference=True
        )
        reference_features = self.prepare_batch(
            filtered_bids,
            filtered_bids,
            is_inference=False,
            query_contract=query_bid.contract,
        )

        with torch.no_grad():
            # checkpoint = Path("models").glob(f"{self.pay_item_id}_*.pt")[0]
            # self.model.load_state_dict(torch.load(checkpoint))
            # self.model.eval()
            self.model = WeightedKNNRegression(k=5)
            prediction = self.model(query_features, reference_features)
            return prediction.item()


if __name__ == "__main__":
    combined_df, val_contracts = combine_csv_files(training=True)
    payitem = "1519000000-E"
    # historic_bids = return_historic_bids_for_payitem(payitem, combined_df)

    # predictor = PayItemPredictor(pay_item_id=payitem)

    # k_results = predictor.train(historic_bids, n_epochs=5)

    for csv in val_contracts:
        csv_filepath = Path(f"nc_csv/{csv}.csv")
        for payitem in loop_contractor_payitems_inference_contract(
            csv_filepath
        ):
            predictor = PayItemPredictor(pay_item_id=payitem.payitem)
            historic_bids = return_historic_bids_for_payitem(
                payitem.payitem, combined_df
            )
            try:
                prediction = predictor.predict(payitem, historic_bids)
            except ValueError as e:
                print(e)
                continue
            print(
                f"{payitem.contractor_id} predicted price for {payitem.payitem}: {prediction}"
            )
            print(f"Grountruth price: {payitem.unit_price}")
            print("\n")
