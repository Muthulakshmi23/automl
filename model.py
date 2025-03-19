# src/model.py
def train_pipeline(pipeline):
    pipeline.fit()

def predict(pipeline):
    predictions = pipeline.predict()
    probs = pipeline.predict_proba()
    return predictions, probs
