
import numpy as np
import retrain_sim as rs


class CARATPolicy:
    """
    Sampled adaptation of CARA-T.

    Relative staleness uses:
      - Current queries
      - Current reference data
      - Data from the last retraining step
      - Predictions of the currently deployed model

    Tuning is performed separately on validation episodes.
    """

    def __init__(
        self,
        episode,
        tau,
        gamma=0.5,
        n_query=64,
        n_data=128,
    ):
        self.ep = episode
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.n_query = n_query
        self.n_data = n_data

        self.last_train = 0

    @staticmethod
    def predict(model, X):
        A = np.column_stack([
            X, np.ones(len(X))
        ])
        return (A @ model >= 0).astype(int)

    def weighted_loss(self, Q, X, Y, model):
        """
        Estimate query-weighted prediction loss.
        """

        pred = self.predict(model, X)
        actual = (Y > 0).astype(int)

        errors = (pred != actual).astype(float)

        # RBF similarity between queries and data
        distances = np.sum(
            (Q[:, None, :] - X[None, :, :]) ** 2,
            axis=2
        )

        similarity = np.exp(
            -self.gamma * distances
        )

        # Average staleness per query
        return float(
            np.mean(similarity @ errors / len(X))
        )

    def __call__(self, obs):
        t = obs["t"]
        last = self.last_train

        # Currently deployed model
        model = rs.model_at(self.ep, last)

        # Current query sample
        Q = self.ep["X"][t][:self.n_query]

        # Current reference data
        X_current = self.ep["Xr"][t][:self.n_data]
        Y_current = self.ep["Yr"][t][:self.n_data]

        # Data used at last retraining
        X_old = self.ep["Xr"][last][:self.n_data]
        Y_old = self.ep["Yr"][last][:self.n_data]

        current_loss = self.weighted_loss(
            Q, X_current, Y_current, model
        )

        old_loss = self.weighted_loss(
            Q, X_old, Y_old, model
        )

        relative_staleness = current_loss - old_loss

        # CARA-T threshold decision
        retrain = int(
            relative_staleness >= self.tau
        )

        if retrain:
            self.last_train = t

        return retrain
