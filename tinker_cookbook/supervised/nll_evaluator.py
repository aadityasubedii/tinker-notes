import itertools

import tinker
from tinker_cookbook.eval.evaluators import TrainingClientEvaluator
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.types import SupervisedDataset


class NLLEvaluator(TrainingClientEvaluator):
    def __init__(self, data: list[tinker.Datum]):
        self.data = data

    async def __call__(self, training_client: tinker.TrainingClient) -> dict[str, float]:
        # forward pass to get logprobs
        future = await training_client.forward_async(self.data, loss_fn="cross_entropy")
        # wait for the result
        result = await future.result_async()
        # extract logprobs from the result
        logprobs = [x["logprobs"] for x in result.loss_fn_outputs]
        # extract weights from the data
        weights = [datum.loss_fn_inputs["weights"] for datum in self.data]
        # compute the mean negative log likelihood
        nll = compute_mean_nll(logprobs, weights)
        return {"nll": nll}

    @classmethod
    def from_dataset(cls, dataset: SupervisedDataset) -> "NLLEvaluator":
        # get all the data from the dataset
        all_data = list(itertools.chain(*[dataset.get_batch(i) for i in range(len(dataset))]))
        return cls(all_data)
