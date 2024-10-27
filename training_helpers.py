import torch
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    print("before")
    print(predictions.shape, labels.shape)

    # Convert numpy arrays to torch tensors
    predictions = torch.tensor(predictions).squeeze()
    labels = torch.tensor(labels)

    print("after")
    print(predictions.shape, labels.shape)

    # Filter out NaN values
    valid_indices = ~torch.isnan(predictions) & ~torch.isnan(labels)
    valid_predictions = predictions[valid_indices]
    valid_labels = labels[valid_indices]

    return {
        "rmse": root_mean_squared_error(valid_labels, valid_predictions),
        "correlation": pearsonr(valid_labels, valid_predictions)[0],
    }
