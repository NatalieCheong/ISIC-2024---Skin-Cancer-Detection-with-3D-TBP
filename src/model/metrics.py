from sklearn.metrics import roc_curve, auc

def calculate_pauc(targets, predictions, min_tpr=0.8):
    fpr, tpr, _ = roc_curve(targets, predictions)
    idx = tpr >= min_tpr
    if any(idx):
        return auc(fpr[idx], tpr[idx])
    return 0.0
