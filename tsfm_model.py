import torch
from chronos import ChronosPipeline

class TSFMForecaster:
    def __init__(self, model_name="amazon/chronos-t5-small"):
        # Φόρτωση του προ-εκπαιδευμένου μοντέλου
        self.pipeline = ChronosPipeline.from_pretrained(model_name)

    def predict(self, context_data):
        """
        context_data: Tensor [n_assets, history_length]
        """
        # Το Chronos κάνει zero-shot forecasting
        forecast = self.pipeline.predict(context_data, prediction_length=1)
        return forecast.mean(dim=1).flatten()