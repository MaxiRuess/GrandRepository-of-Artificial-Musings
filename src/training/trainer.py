import torch
import time
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


def train_model_amp(model, train_dataloader, test_dataloader, epochs, criterion,
                    optimizer, device, scheduler=None):
    """
    Train a model with Automatic Mixed Precision (AMP).

    Auto-detects the best dtype and scaler config for the device.
    Returns results dict with per-epoch losses, accuracies, and timing.
    """
    from src.utils.device import get_amp_config

    model.to(device)
    amp_device_type, amp_dtype, use_scaler = get_amp_config(device)
    scaler = torch.GradScaler(device) if use_scaler else None

    results = {
        "train_loss_per_batch": [],
        "train_loss_per_epoch": [],
        "train_acc_per_epoch": [],
        "val_loss_per_epoch": [],
        "val_acc_per_epoch": [],
        "epoch_times": [],
    }

    for epoch in tqdm(range(epochs)):
        model.train()
        training_loss, training_acc = 0.0, 0.0
        epoch_start = time.perf_counter()

        for batch, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=amp_device_type, dtype=amp_dtype):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            results["train_loss_per_batch"].append(loss.item())
            training_loss += loss.item()
            training_acc += (outputs.argmax(1) == labels).float().mean().item()

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.perf_counter() - epoch_start
        epoch_loss_tr = training_loss / len(train_dataloader)
        epoch_acc_tr = training_acc / len(train_dataloader)

        model.eval()
        test_loss, test_acc = 0.0, 0.0
        with torch.no_grad():
            with torch.autocast(device_type=amp_device_type, dtype=amp_dtype):
                for batch, (inputs, labels) in enumerate(test_dataloader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    test_acc += (outputs.argmax(1) == labels).float().mean().item()

        epoch_loss_te = test_loss / len(test_dataloader)
        epoch_acc_te = test_acc / len(test_dataloader)

        print(f"Epoch {epoch}: train_loss={epoch_loss_tr:.4f}, train_acc={epoch_acc_tr:.4f}, "
              f"val_loss={epoch_loss_te:.4f}, val_acc={epoch_acc_te:.4f}, time={epoch_time:.2f}s")

        results["train_loss_per_epoch"].append(epoch_loss_tr)
        results["train_acc_per_epoch"].append(epoch_acc_tr)
        results["val_loss_per_epoch"].append(epoch_loss_te)
        results["val_acc_per_epoch"].append(epoch_acc_te)
        results["epoch_times"].append(epoch_time)

    return model, results


def train_model_grad_accum(model, train_dataloader, test_dataloader, epochs, criterion,
                           optimizer, device, accumulation_steps=1, use_amp=False,
                           scheduler=None):
    """
    Train a model with gradient accumulation and optional AMP.

    Accumulates gradients over `accumulation_steps` mini-batches before
    calling optimizer.step(), simulating a larger effective batch size.

    Args:
        accumulation_steps: Number of mini-batches to accumulate before stepping.
            Effective batch size = dataloader.batch_size * accumulation_steps.
        use_amp: If True, wrap forward pass in torch.autocast.
    """
    from src.utils.device import get_amp_config

    model.to(device)

    amp_device_type, amp_dtype, use_scaler = None, None, False
    scaler = None
    if use_amp:
        amp_device_type, amp_dtype, use_scaler = get_amp_config(device)
        scaler = torch.GradScaler(device) if use_scaler else None

    results = {
        "train_loss_per_epoch": [],
        "train_acc_per_epoch": [],
        "val_loss_per_epoch": [],
        "val_acc_per_epoch": [],
        "epoch_times": [],
    }

    for epoch in tqdm(range(epochs)):
        model.train()
        training_loss, training_acc = 0.0, 0.0
        epoch_start = time.perf_counter()
        optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            if use_amp:
                with torch.autocast(device_type=amp_device_type, dtype=amp_dtype):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaled_loss = loss / accumulation_steps

            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (i + 1) % accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            training_loss += loss.item()
            training_acc += (outputs.argmax(1) == labels).float().mean().item()

        # Handle trailing batches
        if (i + 1) % accumulation_steps != 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.perf_counter() - epoch_start
        epoch_loss_tr = training_loss / len(train_dataloader)
        epoch_acc_tr = training_acc / len(train_dataloader)

        model.eval()
        test_loss, test_acc = 0.0, 0.0
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type=amp_device_type, dtype=amp_dtype):
                    for batch, (inputs, labels) in enumerate(test_dataloader):
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        test_loss += loss.item()
                        test_acc += (outputs.argmax(1) == labels).float().mean().item()
            else:
                for batch, (inputs, labels) in enumerate(test_dataloader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    test_acc += (outputs.argmax(1) == labels).float().mean().item()

        epoch_loss_te = test_loss / len(test_dataloader)
        epoch_acc_te = test_acc / len(test_dataloader)

        print(f"Epoch {epoch}: train_loss={epoch_loss_tr:.4f}, train_acc={epoch_acc_tr:.4f}, "
              f"val_loss={epoch_loss_te:.4f}, val_acc={epoch_acc_te:.4f}, time={epoch_time:.2f}s")

        results["train_loss_per_epoch"].append(epoch_loss_tr)
        results["train_acc_per_epoch"].append(epoch_acc_tr)
        results["val_loss_per_epoch"].append(epoch_loss_te)
        results["val_acc_per_epoch"].append(epoch_acc_te)
        results["epoch_times"].append(epoch_time)

    return model, results
