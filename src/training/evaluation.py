import torch
from tqdm import tqdm
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix


def evaluate_model(pytorch_model, test_dataloader, device, class_names):
    """Evaluate model and plot confusion matrix."""
    y_preds = []

    pytorch_model.eval()
    with torch.inference_mode():
        for X, y in tqdm(test_dataloader):
            X, y = X.to(device), y.to(device)
            logits = pytorch_model(X)
            pred = torch.argmax(logits, dim=1)
            pred = pred.cpu()
            y_preds.append(pred)

    y_preds = torch.cat(y_preds, dim=0)
    test_truth = torch.cat([y for _, y in test_dataloader], dim=0)

    confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass").to(device)
    confmat_tensor = confmat(y_preds.to(device), test_truth.to(device))
    confmat_tensor = confmat_tensor.cpu()

    fig, ax = plot_confusion_matrix(confmat_tensor.numpy(), figsize=(10, 10),
                                     class_names=class_names, show_normed=True)
    fig.show()

    return y_preds, test_truth
