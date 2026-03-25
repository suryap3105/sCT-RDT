import sklearn.metrics as metrics

def calculate_metrics(predictions, targets):
    accuracy = metrics.accuracy_score(targets, predictions)
    macro_f1 = metrics.f1_score(targets, predictions, average='macro', zero_division=0)
    # Use macro to treat rare Supernovae equally to common background stars
    
    return accuracy, macro_f1
