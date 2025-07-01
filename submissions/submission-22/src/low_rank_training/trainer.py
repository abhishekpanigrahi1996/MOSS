import torch
import torch.nn as nn
import torch.optim as optim
import csv
import time
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        scheduler,
        optimizer,
        device,
        lr_layers,
        args=None,
    ):
        """Constructs trainer which manages and trains neural network
        Args:
            net_architecture: Dictionary of the network architecture. Needs keys 'type' and 'dims'. Low-rank layers need key 'rank'.
            train_loader: loader for training data
            test_loader: loader for test data
        """
        # Set the device (GPU or CPU)
        self.device = device

        # Initialize the model
        self.model = model
        # store train and test data
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.best_accuracy = 0
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.lr_layers = lr_layers
        print(self.lr_layers)
        self.args = args

    def train(
        self,
    ):

        if self.args.wandb:
            import wandb

            watermark = "model-{}_data-{}".format(
                self.args.model,
                self.args.pretrained_weights,
            )
            wandb.init(
                project="model-{}_data-{}".format(
                    self.args.model,
                    self.args.pretrained_weights,
                ),
            )
            wandb.config.update(self.args)
            wandb.watch(self.model)

        # Define the loss function and optimizer. Optimizer is only needed to set all gradients to zero.
        criterion = nn.CrossEntropyLoss()

        # Training loop
        k = len(self.train_loader)
        # Use DataParallel to utilize both GPUs
        if self.args.multi_gpu:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)  # Wrap the model
        self.model.to(self.device)

        for epoch in range(self.args.epochs):
            self.model.train()
            loss_hist = 0
            epoch_start_time = time.time()
            with tqdm(
                total=len(self.train_loader),
                desc=f"Epoch {epoch}",
            ) as pbar:
                for batch_idx, (data, targets) in enumerate(self.train_loader):
                    data = data.to(self.device)
                    targets = targets.to(self.device)

                    # Forward pass
                    outputs = self.model(data)
                    loss = criterion(outputs, targets)

                    # Backward
                    self.optimizer.zero_grad()
                    loss.backward()

                    # ----- DLRT -----
                    if batch_idx % self.args.num_local_iter == 1:
                        for layer in self.lr_layers:
                            layer.augment()

                    else:
                        for layer in self.lr_layers:
                            layer.set_basis_grad_zero()
                        self.optimizer.step()
                        if batch_idx % self.args.num_local_iter == 0:
                            for layer in self.lr_layers:
                                layer.truncate()
                    # ----- DLRT -----

                    loss_hist += float(loss.item()) / (k * self.args.batch_size)

                    pbar.set_description(
                        f"Epoch [{epoch + 1}/{self.args.epochs}], Loss: {loss.item():.4f}, Ranks: { [lr_layer.r for lr_layer in self.lr_layers]}"
                    )
                    pbar.update(1)

            # Test the model
            elapsed = time.time() - epoch_start_time
            # evaluate model on test date
            self.scheduler.step()
            curr_lr = self.optimizer.param_groups[0]["lr"]  # update learning rate

            val_loss, accuracy, cr = self.test_model(
                elapsed, epoch, train_loss=float(loss.item()), curr_lr=curr_lr
            )
            if self.args.wandb:
                wandb.log(
                    {
                        "loss train": float(loss.item()),
                        "loss_val": val_loss,
                        "val_accuracy": accuracy,
                        "best val acc": self.best_accuracy,
                        "learning_rate": curr_lr,
                        "compression": cr,
                        "rank ": [lr_layer.r for lr_layer in self.lr_layers],
                    }
                )
            # adapt learning rate
            # self.scheduler.step(epoch)
            torch.save(self.model.state_dict(), "tmp_model.pt")

        print("Training finished.")

    @torch.no_grad()
    def test_model(self, elaped_time, epoch, train_loss, curr_lr):
        """Prints the model's accuracy on the test data"""
        # Test the model
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, targets in self.test_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                val_loss = nn.CrossEntropyLoss()(outputs, targets)

            accuracy = 100 * correct / total
            print(f"Accuracy of the network on the test images: {accuracy}%")
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
        cr = self.compute_compression()
        print("-" * 110)
        print(
            f"| end of epoch {epoch:3d} | time: {elaped_time:5.2f}s | "
            f"valid loss {val_loss.cpu().numpy():5.2f} | valid acc {accuracy:8.2f} | best val acc {self.best_accuracy:8.2f} | "
            f"c.r.: {cr:5.2f} | lr: {curr_lr}"
        )  # | ranks: {self.model.get_ranks()}
        print("-" * 110)

        return val_loss.cpu().numpy(), accuracy, cr

    def compute_compression(self):
        full_params = 0
        lr_prams = 0
        for layer in self.lr_layers:
            lr_prams += (
                layer.U.shape[0] * layer.r + layer.r**2 + layer.V.shape[0] * layer.r
            )
            full_params += layer.U.shape[0] * layer.V.shape[0]

            cr = (1 - lr_prams / full_params) * 100
        return cr
