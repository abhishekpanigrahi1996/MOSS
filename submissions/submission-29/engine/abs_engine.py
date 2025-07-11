import os

import torch.nn.functional
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import log_loss
from abc import ABC, abstractmethod
import plot.curve as curve
from sklearn.metrics import accuracy_score
import json

from model.device_check import *


def seed_worker(worker_id):
    # worker_seed = torch.initial_seed() % 2 ** 32
    # np.random.seed(worker_seed)
    # random.seed(worker_seed)
    seed = 42
    np.random.seed(int(seed))


class EncoderClassifier:
    def __init__(self, cfg, result_path):
        # load the data set
        self.result_path = result_path

        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.init_dataset(cfg["data"])
        self.verbose = True

        # load the model
        self.encoder = None  # this is liunet
        self.classifier = None  # this is a simple multilayer perceptron to classify the output of encoder
        self.optim = None
        self.loss_func = None
        self.init_models(cfg["train"])
        self.cfg = cfg

    def set_train(self):
        for m in self.trained_models:
            getattr(self, m).train()

    def set_eval(self):
        for m in self.trained_models:
            getattr(self, m).eval()

    @abstractmethod
    def init_dataset(self, cfg_data):
        pass

    @abstractmethod
    def init_models(self, cfg_train):
        # init the encoder
        # init the classifier
        # init the loss function
        # init the optimizer
        pass

    def is_best_model(self, current, current_best, hist=None):
        return current < current_best

    def save_model(self, model_names, criteria, epoch_num):
        state = {
            'epoch': epoch_num,
            'optimizer': self.optim.state_dict(),
            "eval": {}
        }

        for k in criteria:
            state["eval"][k] = criteria[k]

        for k in model_names:
            state[k] = getattr(self, k).state_dict()

        torch.save(state, self.result_path + '/model.pth.tar')

    def load_model(self, checkpoint_path):
        d = torch.load(checkpoint_path)
        for key in d:
            if key not in ["epoch", "optimizer", "eval"]:
                getattr(self, key).load_state_dict(d[key])

        return 'best epoch: {}, eval: {}\n'.format(d['epoch'], d['eval'])

    @abstractmethod
    def forward_pass(self, input_tuple):
        pass

    def train(self, cfg_train):
        """
        train & evaluate & plot
        """
        best_eval = np.inf
        losses = []
        val_ce_losses = []
        val_accs = []
        
        train_subset = self.train_set.select(range(self.subsets))

        train_loader = torch.utils.data.DataLoader(train_subset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   # num_workers=2,
                                                   worker_init_fn=seed_worker)

        val_loader = torch.utils.data.DataLoader(self.val_set,
                                                 batch_size=self.val_batch_size,
                                                 shuffle=True,
                                                 # num_workers=2,
                                                 worker_init_fn=seed_worker)

        for i in tqdm(range(int(cfg_train['max_epoch']))):
            self.set_train()

            loss_epoch = 0

            # train the model
            for batch_idx, input_tuple in enumerate(train_loader):
                self.optim.zero_grad()

                print(input_tuple)

                # input_tuple = (input_tuple['img'], input_tuple['text'], input_tuple['label'], input_tuple['value'])

                output, labels = self.forward_pass(input_tuple)

                loss = self.loss_func(output, labels)

                loss.backward()
                self.optim.step()

                loss_epoch += loss.item()

            losses.append(loss_epoch / len(train_loader))

            # validate the model
            self.encoder.eval()

            if self.classifier is not None:
                self.classifier.eval()

            val_ce_loss, val_acc = self.evaluate(val_loader)
            _, train_acc = self.evaluate(train_loader)

            val_ce_losses.append(val_ce_loss)
            val_accs.append(val_acc)

            if self.is_best_model(val_ce_loss, best_eval):
                model_dict = {"encoder": self.encoder}
                if self.classifier is not None:
                    model_dict["classifier"] = self.classifier
                self.save_model(
                    model_dict,
                    {"ce_loss": val_ce_loss},
                    i
                )
                best_eval = val_ce_loss
            # print("train loss: {}, val loss: {}, val accuracy: {}".format(loss_epoch / len(train_loader),
            #                                                               val_ce_loss,
            #                                                               val_acc))

            print("train loss: {}, train accuracy: {}, val loss: {}, val accuracy: {}".format(
                loss_epoch / len(train_loader),
                train_acc,
                val_ce_loss,
                val_acc))

            # plot training losses and validation ce loss up to the current epoch
            curve.single_plot_one_curve(np.arange(i + 1), losses, "epoch", "ce_loss",
                                        self.result_path + "/train_losses.png")
            curve.single_plot_one_curve(np.arange(i + 1), val_ce_losses, "epoch", "ce_loss",
                                        self.result_path + "/val_losses.png")
            curve.single_plot_one_curve(np.arange(i + 1), val_accs, "epoch", "accuracy",
                                        self.result_path + "/val_accuracy.png")

    def evaluate(self, loader, save_json=False, json_filename='predictions.json'):
        # compute the overall ce_loss on the data loader using sklearn log_loss function
        self.set_eval()

        ground_truth = []
        predict = []

        for batch_idx, input_tuple in enumerate(loader):
            # input_tuple = (input_tuple['img'], input_tuple['text'], input_tuple['label'], input_tuple['value'])
            output, labels = self.forward_pass(input_tuple)

            ground_truth.append(labels.long().cpu().data.numpy())
            predict.append(output.cpu().data.numpy())

        # Concatenate the lists to form arrays
        ground_truth = np.concatenate(ground_truth, axis=0)
        predict = np.concatenate(predict, axis=0)

        # Save predictions and ground truth to JSON if requested
        if save_json:
            # Convert numpy arrays to lists for JSON serialization
            data_to_save = {
                'ground_truth': ground_truth.tolist(),
                'predict': predict.tolist()
            }
            with open(json_filename, 'w') as f:
                json.dump(data_to_save, f)

        ce_loss = log_loss(ground_truth, predict)

        predict_labels = np.argmax(predict, axis=1)
        ground_truth_labels = np.argmax(ground_truth, axis=1)
        accuracy = accuracy_score(ground_truth_labels, predict_labels)
        return ce_loss, accuracy

    def test(self):
        # load the trained model and test it on the test set
        cp_state = self.load_model(
            self.result_path + '/model.pth.tar'
        )

        self.encoder.eval()

        if self.classifier is not None:
            self.classifier.eval()

        print("#" * 10, "trained model loaded")
        print(cp_state)

        test_loader = torch.utils.data.DataLoader(self.test_set,
                                                  batch_size=self.val_batch_size,
                                                  shuffle=False,
                                                  # num_workers=2
                                                  worker_init_fn=seed_worker,
                                                # collate_fn=self.collate
                                                 )
        return self.evaluate(test_loader, True, os.path.join(self.result_path, self.cfg["exp_name"]+"_test_results.json"))
        
    def collate(self, batch):
        bzs = len(batch)
        img = torch.tensor([batch[i]['img'] for i in range(bzs)])
        text = torch.tensor([batch[i]['text'] for i in range(bzs)])
        label = torch.tensor([batch[i]['label'] for i in range(bzs)])
        value = torch.tensor([batch[i]['value'] for i in range(bzs)])
        return (img, text, label, value)

class EncoderClassifierEarlyStop(EncoderClassifier, ABC):
    def __init__(self, cfg, result_path):
        self.loss_window_size = 10
        self.loss_std_threshold = 1e-2

        super().__init__(cfg, result_path)

    def is_best_model(self, current, current_best, hist=None):
        return current > current_best        
    
    def collate(self, batch):
        bzs = len(batch)
        img = torch.tensor([batch[i]['img'] for i in range(bzs)])
        text = torch.tensor([batch[i]['text'] for i in range(bzs)])
        label = torch.tensor([batch[i]['label'] for i in range(bzs)])
        value = torch.tensor([batch[i]['value'] for i in range(bzs)])
        return (img, text, label, value)
        
    def train(self, cfg_train):
        """
        Train, evaluate, and plot while displaying training loss in the progress bar.
        """
        best_eval = 0
        losses = []
        val_ce_losses = []
        val_accs = []

        # train_subset = self.train_set.select(range(self.subsets))

        train_loader = torch.utils.data.DataLoader(self.train_set,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   worker_init_fn=seed_worker,
                                                   # collate_fn=self.collate
                                                  )

        val_loader = torch.utils.data.DataLoader(self.val_set,
                                                 batch_size=self.val_batch_size,
                                                 shuffle=True,
                                                 worker_init_fn=seed_worker,
                                                 # collate_fn=self.collate
                                                )

        # Create a tqdm progress bar and update its description with the training loss
        for i in tqdm(range(int(cfg_train['max_epoch'])), desc="Training Epochs"):
            # print(i / int(cfg_train['max_epoch']))
            self.set_train()

            loss_epoch = 0

            # train the model
            for batch_idx, input_tuple in enumerate(train_loader):
                self.optim.zero_grad()

                # print(input_tuple)
                # for el in input_tuple:
                #     print(el.shape)

                # input_tuple = (input_tuple['img'], input_tuple['text'], input_tuple['label'], input_tuple['value'])

                output, labels = self.forward_pass(input_tuple)
                # print(output.type())
                # print(labels.type())
                loss = self.loss_func(output, labels)

                loss.backward()
                self.optim.step()

                loss_epoch += loss.item()

            avg_train_loss = loss_epoch / len(train_loader)
            losses.append(avg_train_loss)

            # Update tqdm progress bar description with the training loss
            tqdm.write(f"Epoch {i+1}/{cfg_train['max_epoch']}, Train Loss: {avg_train_loss:.4f}")

            # Validate the model
            self.encoder.eval()
            if self.classifier is not None:
                self.classifier.eval()

            val_ce_loss, val_acc = self.evaluate(val_loader)
            _, train_acc = self.evaluate(train_loader)

            val_ce_losses.append(val_ce_loss)
            val_accs.append(val_acc)

            if self.is_best_model(val_acc, best_eval):
                model_dict = {"encoder": self.encoder}
                if self.classifier is not None:
                    model_dict["classifier"] = self.classifier
                self.save_model(
                    model_dict,
                    {"val_accuracy": val_acc},
                    i
                )
                best_eval = val_acc

            # Print detailed progress if verbose
            if self.verbose:
                print(f"Epoch {i+1}/{cfg_train['max_epoch']}, Train Loss: {avg_train_loss:.4f}, "
                      f"Train Accuracy: {train_acc:.4f}, Val Loss: {val_ce_loss:.4f}, Val Accuracy: {val_acc:.4f}")

            # Plot training and validation metrics
            curve.single_plot_one_curve(np.arange(i + 1), losses, "epoch", "ce_loss",
                                        self.result_path + "/train_losses.png")
            curve.single_plot_one_curve(np.arange(i + 1), val_ce_losses, "epoch", "ce_loss",
                                        self.result_path + "/val_losses.png")
            curve.single_plot_one_curve(np.arange(i + 1), val_accs, "epoch", "accuracy",
                                        self.result_path + "/val_accuracy.png")
            
            if type(losses) != list:
                losses = losses.tolist()
            if type(val_ce_losses) != list:
                val_ce_losses = val_ce_losses.tolist()
            if type(val_accs) != list:
                val_accs = val_accs.tolist()
            
            # Collect data to save to JSON
            metrics_data = {
                "epochs": np.arange(i + 1).tolist(),  # Convert NumPy array to list
                "losses": losses,
                "val_ce_losses": val_ce_losses,
                "val_accs": val_accs
            }

            # Define the output path for the JSON file
            json_file_path = self.result_path + "/training_metrics.json"

            # Write data to a JSON file
            with open(json_file_path, 'w') as json_file:
                json.dump(metrics_data, json_file, indent=4)

            if i + 1 > self.loss_window_size and self.converge(losses[-self.loss_window_size:]):
                break

    def converge(self, val_losses):
        return torch.std(torch.tensor(val_losses)) <= self.loss_std_threshold
