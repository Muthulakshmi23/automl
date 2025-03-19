# src/fusion.py
from autoprognosis import AutoMLPipeline
from autoprognosis.explorers.core.defaults import default_image_estimators, default_tabular_estimators

def build_pipeline(image_features, tabular_features, labels_encoded):
    pipeline = AutoMLPipeline(
        image_data=image_features,
        tabular_data=tabular_features,
        targets=labels_encoded,
        image_estimators=default_image_estimators,
        tabular_estimators=default_tabular_estimators,
        fusion_mode="auto"
    )
    return pipeline
