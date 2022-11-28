from sklearn import metrics as sklearn_metrics

def compute_metrics(labels, preds):
    return {
        "precision": sklearn_metrics.precision_score(labels, preds, average="macro"),
        "recall": sklearn_metrics.recall_score(labels, preds, average="macro"),
        "f1": sklearn_metrics.f1_score(labels, preds, average="macro"),
        "acc": sklearn_metrics.accuracy_score(labels, preds)
    }

if __name__ == '__main__':
    a = [1, 1, 2, 3, 1, 1, 1, 2, 2, 2]
    b = [1, 1, 1, 3, 1, 1, 1, 2, 1, 1]

    d = compute_metrics(a, b)

    print()