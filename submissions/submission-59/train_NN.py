import torch
from tqdm import tqdm
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_classifier(
    model: nn.Module,
    train_data_loader: torch.utils.data.DataLoader,
    test_data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    label_column_name: str,
    num_epochs: int = 5,
    evaluation_interval: int = 5,
):
    train_losses = []
    train_accuracies = []

    eval_losses = []
    eval_accuracies = []
    eval_indexes = []

    model = model.to(device)
    model.train()

    print(f"Starting training on device: {device}")

    for epoch in (bar := tqdm(range(num_epochs), desc="Epochs")):
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        # model.freeze_encoder() # NoEncoder has no params

        for i, batch in enumerate(train_data_loader):
            try:
                input_embeddings = batch["images"].to(device, non_blocking=True)
                target = batch[label_column_name].to(device, non_blocking=True)

                batch_on_device = {"images": input_embeddings}
            except KeyError as e:
                print(f"\nError: Batch dictionary missing expected key: {e} in training batch {i}. Skipping batch.")
                print(f"Batch keys: {batch.keys()}")
                continue
            except Exception as e:
                print(f"\nError moving batch {i} to device: {e}. Skipping batch.")
                continue

            optimizer.zero_grad()

            predicted = model(batch_on_device)
            loss = criterion(predicted, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted_labels = torch.max(predicted.detach(), 1)
            correct_train += (predicted_labels == target).sum().item()
            total_train += target.size(0)

        average_epoch_loss = total_loss / len(train_data_loader) if len(train_data_loader) > 0 else 0.0
        train_losses.append(average_epoch_loss)

        train_epoch_accuracy = correct_train / total_train if total_train > 0 else 0.0
        train_accuracies.append(train_epoch_accuracy)

        if epoch % evaluation_interval == 0 or epoch == num_epochs - 1:
            eval_loss, eval_accuracy = eval_classifier(
                model=model,
                test_data_loader=test_data_loader,
                criterion=criterion,
                label_column_name=label_column_name,
            )
            eval_losses.append(eval_loss)
            eval_accuracies.append(eval_accuracy)
            eval_indexes.append(epoch)
            model.train()
            bar.set_description(
                f"Epoch {epoch}/{num_epochs-1}, Train Loss: {average_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.4f}, Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_accuracy:.4f}"
            )
        else:
            bar.set_description(
                f"Epoch {epoch}/{num_epochs-1}, Train Loss: {average_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.4f}"
            )

    return train_losses, eval_losses, train_accuracies, eval_accuracies, eval_indexes


def eval_classifier(
    model: nn.Module,
    test_data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    label_column_name: str,
):

    model.eval()
    total_loss = 0.0
    correct_eval = 0
    total_eval = 0

    with torch.no_grad():
        for i, batch in enumerate(test_data_loader):
            try:
                input_embeddings = batch["images"].to(device, non_blocking=True)
                target = batch[label_column_name].to(device, non_blocking=True)

                batch_on_device = {"images": input_embeddings}
            except KeyError as e:
                print(f"\nError: Batch dictionary missing expected key: {e} in evaluation batch {i}. Skipping batch.")
                print(f"Batch keys: {batch.keys()}")
                continue
            except Exception as e:
                print(f"\nError moving batch {i} to device: {e}. Skipping batch.")
                continue

            predicted = model(batch_on_device)

            loss = criterion(predicted, target)

            total_loss += loss.item()

            _, predicted_labels = torch.max(predicted, 1)
            correct_eval += (predicted_labels == target).sum().item()
            total_eval += target.size(0)

    average_eval_loss = total_loss / len(test_data_loader) if len(test_data_loader) > 0 else 0.0
    eval_accuracy = correct_eval / total_eval if total_eval > 0 else 0.0

    return average_eval_loss, eval_accuracy
