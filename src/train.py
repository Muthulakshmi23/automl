# src/train.py
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate(y_true, y_pred, probs):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred, average='weighted'))
    print("AUC-ROC:", roc_auc_score(y_true, probs, multi_class='ovr'))
