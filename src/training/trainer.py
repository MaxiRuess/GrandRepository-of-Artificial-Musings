import torch
from tqdm import tqdm


def train_model(model, train_dataloader, test_dataloader, epochs, criterion, optimizer, device, scheduler=None):
    """
    Train a model and return results dict with per-epoch losses and accuracies.
    """
    model.to(device)

    results = {
        "train_loss_per_batch": [],
        "train_loss_per_epoch": [],
        "train_acc_per_epoch": [],
        "val_loss_per_epoch": [],
        "val_acc_per_epoch": [],
    }

    for epoch in tqdm(range(epochs)):
        model.train()
        training_loss, training_acc = 0.0, 0.0

        for batch, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            results["train_loss_per_batch"].append(loss.item())
            training_loss += loss.item()
            training_acc += (outputs.argmax(1) == labels).float().mean()
            training_acc = training_acc.item()

        if scheduler is not None:
            scheduler.step()

        epoch_loss_tr = training_loss / len(train_dataloader)
        epoch_acc_tr = training_acc / len(train_dataloader)
        print(f"Epoch: {epoch} Training Loss: {epoch_loss_tr}, Training Accuracy: {epoch_acc_tr}")

        model.eval()
        test_loss, test_acc = 0.0, 0.0

        for batch, (inputs, labels) in enumerate(test_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_acc += (outputs.argmax(1) == labels).float().mean()
            test_acc = test_acc.item()

        epoch_loss_te = test_loss / len(test_dataloader)
        epoch_acc_te = test_acc / len(test_dataloader)
        print(f"Epoch: {epoch} Validation Loss: {epoch_loss_te}, Validation Accuracy: {epoch_acc_te}")

        results["train_loss_per_epoch"].append(epoch_loss_tr)
        results["train_acc_per_epoch"].append(epoch_acc_tr)
        results["val_loss_per_epoch"].append(epoch_loss_te)
        results["val_acc_per_epoch"].append(epoch_acc_te)

    return model, results
