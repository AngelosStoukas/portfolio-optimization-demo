import torch
from chronos import ChronosPipeline


class TSFMForecaster:
    def __init__(self, model_name="amazon/chronos-t5-small"):
        print(f"Φόρτωση {model_name}...")
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float32
        )
        print("✅ Chronos φορτώθηκε!")

    def predict(self, context_data):
        # context_data: [n_assets, history_length]
        # forecast shape: [n_assets, num_samples, 1]
        forecast = self.pipeline.predict(
            context=context_data,
            prediction_length=1
        )
        # → mean over samples → [n_assets, 1] → squeeze → [n_assets]
        return forecast.mean(dim=1).squeeze(-1)
