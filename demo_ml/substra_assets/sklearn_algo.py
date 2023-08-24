from substrafl import algorithms
from substrafl import remote
from substrafl.strategies import schemas as fl_schemas

import numpy as np
import joblib
from typing import Optional
import shutil

# The dataset proposes four attributes to predict three different classes.
INPUT_SIZE = 4
OUTPUT_SIZE = 1


class SklearnLogisticRegression(algorithms.Algo):
    def __init__(self, model, seed=None):
        super().__init__(model=model, seed=seed)

        self._model = model

        # We need all different instances of the algorithm to have the same
        # initialization.
        self._model.coef_ = np.ones((OUTPUT_SIZE, INPUT_SIZE))
        self._model.intercept_ = np.zeros(3)
        self._model.classes_ = np.array([-1])

        if seed is not None:
            np.random.seed(seed)

    @property
    def strategies(self):
        """List of compatible strategies"""
        return [fl_schemas.StrategyName.FEDERATED_AVERAGING]

    @property
    def model(self):
        return self._model

    @remote.remote_data
    def train(
        self,
        datasamples,
        shared_state: Optional[fl_schemas.FedAvgAveragedState] = None,
    ) -> fl_schemas.FedAvgSharedState:
        """The train function to be executed on organizations containing
        data we want to train our model on. The @remote_data decorator is mandatory
        to allow this function to be sent and executed on the right organization.

        Args:
            datasamples: datasamples extracted from the organizations data using
                the given opener.
            shared_state (Optional[fl_schemas.FedAvgAveragedState], optional):
                shared_state provided by the aggregator. Defaults to None.

        Returns:
            fl_schemas.FedAvgSharedState: State to be sent to the aggregator.
        """

        if shared_state is not None:
            # If we have a shared state, we update the model parameters with
            # the average parameters updates.
            self._model.coef_ += np.reshape(
                shared_state.avg_parameters_update[:-1],
                (OUTPUT_SIZE, INPUT_SIZE),
            )
            self._model.intercept_ += shared_state.avg_parameters_update[-1]

        # To be able to compute the delta between the parameters before and after training,
        # we need to save them in a temporary variable.
        old_coef = self._model.coef_
        old_intercept = self._model.intercept_

        # Model training.
        self._model.fit(datasamples["data"], datasamples["targets"])

        # We compute de delta.
        delta_coef = self._model.coef_ - old_coef
        delta_bias = self._model.intercept_ - old_intercept

        # We reset the model parameters to their state before training in order to remove
        # the local updates from it.
        self._model.coef_ = old_coef
        self._model.intercept_ = old_intercept

        # We output the length of the dataset to apply a weighted average between
        # the organizations regarding their number of samples, and the local
        # parameters updates.
        # These updates are sent to the aggregator to compute the average
        # parameters updates, that we will receive in the next round in the
        # `shared_state`.
        return fl_schemas.FedAvgSharedState(
            n_samples=len(datasamples["targets"]),
            parameters_update=[p for p in delta_coef] + [delta_bias],
        )

    @remote.remote_data
    def predict(self, datasamples, shared_state, predictions_path):
        """The predict function to be executed on organizations containing
        data we want to test our model on. The @remote_data decorator is mandatory
        to allow this function to be sent and executed on the right organization.

        Args:
            datasamples: datasamples extracted from the organizations data using
                the given opener.
            shared_state: shared_state provided by the aggregator.
            predictions_path: Path where to save the predictions.
                This path is provided by Substra and the metric will automatically
                get access to this path to load the predictions.
        """
        predictions = self._model.predict(datasamples["data"])

        if predictions_path is not None:
            np.save(predictions_path, predictions)

            # np.save() automatically adds a ".npy" to the end of the file.
            # We rename the file produced by removing the ".npy" suffix, to make sure that
            # predictions_path is the actual file name.
            shutil.move(str(predictions_path) + ".npy", predictions_path)

    def save_local_state(self, path):
        joblib.dump(
            {
                "model": self._model,
                "coef": self._model.coef_,
                "bias": self._model.intercept_,
            },
            path,
        )

    def load_local_state(self, path):
        loaded_dict = joblib.load(path)
        self._model = loaded_dict["model"]
        self._model.coef_ = loaded_dict["coef"]
        self._model.intercept_ = loaded_dict["bias"]
        return self
