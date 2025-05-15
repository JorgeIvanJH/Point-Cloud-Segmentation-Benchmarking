import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import numpy as np
import time

from storage_utils import save_statistics
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from collections import Counter

matplotlib.rcParams.update({"font.size": 8})


class ExperimentBuilder:
    def __init__(
        self,
        model,
        experiment_name,
        num_epochs,
        train_data,
        val_data,
        test_data,
        device: torch.device,
        continue_from_epoch,
        optimizer,
        scheduler,
        loss_criterion,
        metrics: dict,
    ):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param model: A pytorch nn.Module which implements a network architecture.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param device: device to use for training. Can be either "cpu" or "cuda". If cuda is available, then the model will be sent to the GPU.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        :param optimizer: An optimizer to use for training. This is a pytorch optimizer.
        :param scheduler: A learning rate scheduler to use for training. This is a pytorch scheduler.
        :param loss_criterion: A loss function to use for training. This is a pytorch loss function.
        :param metrics: A dictionary of metrics to use for tracking performance. The keys are the names of the metrics and the values are the functions that compute the metrics.
        """
        super(ExperimentBuilder, self).__init__()

        self.experiment_name = experiment_name
        self.num_epochs = num_epochs
        self.model = model
        self.device = device
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_criterion = loss_criterion
        self.metrics = metrics
        print("Selected device: ",self.device)


        # Generate the directory names
        self.experiment_folder = os.path.abspath("experiments/" + experiment_name)
        self.experiment_logs = os.path.abspath(
            os.path.join(self.experiment_folder, "result_outputs")
        )
        self.experiment_saved_models = os.path.abspath(
            os.path.join(self.experiment_folder, "saved_models")
        )
        if not os.path.exists(
            self.experiment_folder
        ):
            os.mkdir(self.experiment_folder)
            os.mkdir(self.experiment_logs)
            os.mkdir(
                self.experiment_saved_models
            )

            
        self.lowest_val_loss_model_idx = 0
        self.lowest_val_loss_model_value = 0.0

        # Training beginning
        if (
            continue_from_epoch == -2
        ):  # if continue from epoch is -2 then continue from latest saved model
            self.state, self.lowest_val_loss_model_idx, self.lowest_val_loss_model_value = (
                self.load_model(
                    model_save_dir=self.experiment_saved_models,
                    model_save_name="train_model",
                    model_idx="latest",
                )
            )  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = int(self.state["model_epoch"])

        elif continue_from_epoch > -1:  # if continue from epoch is greater than -1 then
            self.state, self.lowest_val_loss_model_idx, self.lowest_val_loss_model_value = (
                self.load_model(
                    model_save_dir=self.experiment_saved_models,
                    model_save_name="train_model",
                    model_idx=continue_from_epoch,
                )
            )  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = continue_from_epoch
        else:
            self.state = dict()
            self.starting_epoch = 0


    def plot_func_def(self, all_grads, layers, epoch):
        """
        Plot function definition to plot the average gradient with respect to the number of layers in the given model
        :param all_grads: Gradients wrt weights for each layer in the model.
        :param layers: Layer names corresponding to the model parameters
        :return: plot for gradient flow
        """
        
        colormap = matplotlib.cm.get_cmap("viridis") 
        color = colormap(epoch / self.num_epochs) 
        plt.plot(all_grads, alpha=0.7, color=color)
        plt.hlines(0, 0, len(all_grads) + 1, linewidth=1, color="k")
        plt.xticks(range(0, len(all_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(all_grads))
        plt.xlabel("Layers")
        plt.ylabel("Average Gradient")
        plt.title("Gradient Flow")
        plt.grid(True)
        plt.tight_layout()

        return plt

    def plot_grad_flow(self, named_parameters):
        """
        This function takes as
        input the model parameters during training, accumulates the absolute mean of the gradients in all_grads and
        the layer names in layers. The matplotlib function plt plots gradient values for each layer and the function
        plot_grad_flows() returns this final plot

        """
        all_grads = []
        layers = []
        for name, param in named_parameters:
            if param.requires_grad and param.grad is not None and "bias" not in name:
                # Compute the absolute mean of the gradient
                grad_mean = param.grad.abs().mean().item()
                all_grads.append(grad_mean)
                modified_name = (
                    name.replace("layer_dict.", "")
                    .replace(".weight", "")
                    .replace(".", "_")
                )
                layers.append(modified_name)
        epoch = self.current_epoch
        plt = self.plot_func_def(all_grads, layers,epoch)

        return plt

    def run_iter(self, points, targets, train=True):
        """
        Runs either a training or evaluation iteration according to 'train'
        """
        # Move to device
        points = points.transpose(2, 1).float().to(self.device)  # (B, C, N)
        targets = torch.argmax(targets, dim=2).long().to(self.device)  # Convert from one-hot to class indices

        if train:
            self.model.train()
            preds, _ = self.model(points)  # shape: (B, N, num_classes)
            pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)
            loss = self.loss_criterion(preds, targets, pred_choice)   
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        else:
            self.model.eval()
            with torch.no_grad():
                preds = self.model(points)  # shape: (B, N, num_classes)
                pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)
                loss = self.loss_criterion(preds, targets, pred_choice)

        # Compute metrics
        performance_metrics = {
            name: fn(pred_choice, targets) for name, fn in self.metrics.items()
        }

        return loss.item(), performance_metrics

    def save_model(
        self,
        model_save_dir,
        model_save_name,
        model_idx,
        lowest_val_loss_model_idx,
        lowest_val_loss_model_value,
    ):
        """
        Save the network parameter state and current lowest val loss with epoch idx.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param lowest_val_loss_model_idx: The index of the model with lowest loss to be stored for future use.
        :param lowest_val_loss_model_value: The lowest validation loss to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        self.state["network"] = (
            self.state_dict()
        )  # save network parameter and other variables.
        self.state["lowest_val_loss_model_idx"] = (
            lowest_val_loss_model_idx  # save current lowest val loss idx
        )
        self.state["lowest_val_loss_model_value"] = (
            lowest_val_loss_model_value  # save current lowest val loss
        )
        torch.save(
            self.state,
            f=os.path.join(
                model_save_dir, "{}_{}".format(model_save_name, str(model_idx))
            ),
        )  # save state at prespecified filepath

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        state = torch.load(
            f=os.path.join(
                model_save_dir, "{}_{}".format(model_save_name, str(model_idx))
            )
        )
        return state, state["lowest_val_loss_model_idx"], state["lowest_val_loss_model_value"]

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        ruonceflag=True
        total_losses = {} # initialize a dict to keep the per-epoch metrics
        total_losses["train_loss"] = []
        total_losses["val_loss"] = []
        total_losses.update({f"train_{metric_name}": [] for metric_name in self.metrics.keys()})
        total_losses.update({f"val_{metric_name}": [] for metric_name in self.metrics.keys()})

        self.lowest_val_loss_model_value = float("inf")  # set the lowest val loss to infinity
        self.lowest_val_loss_model_idx = 0  # set the lowest val loss model idx to 0

        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            epoch_start_time = time.time()

            current_epoch_losses = {}
            current_epoch_losses["train_loss"] = []
            current_epoch_losses["val_loss"] = []
            current_epoch_losses.update({f"train_{metric_name}": [] for metric_name in self.metrics.keys()})
            current_epoch_losses.update({f"val_{metric_name}": [] for metric_name in self.metrics.keys()})
            self.current_epoch = epoch_idx
            
            # Run training and validation iterations
            for phase, dataloader in [("train", self.train_data), ("val", self.val_data)]:
                is_train = phase == "train"
                with tqdm.tqdm(total=len(dataloader), desc=phase) as pbar:
                    for points, targets in dataloader:
                        loss, performance_metrics = self.run_iter(points, targets, train=is_train)
                        current_epoch_losses[f"{phase}_loss"].append(loss)

                        for metric_name, metric_value in performance_metrics.items():
                            current_epoch_losses[f"{phase}_{metric_name}"].append(metric_value)

                        pbar.update(1)
                        pbar.set_postfix(loss=f"{loss:.4f}")

            
            val_mean_loss = np.mean(current_epoch_losses["val_loss"])
            if (val_mean_loss < self.lowest_val_loss_model_value):
                self.lowest_val_loss_model_value = val_mean_loss 
                self.lowest_val_loss_model_idx = epoch_idx

            # mean of all metrics for storage and output on the terminal.
            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(value)) 
            save_statistics(
                experiment_log_dir=self.experiment_logs,
                filename="summary.csv",
                stats_dict=total_losses,
                current_epoch=self.current_epoch,
                continue_from_mode=(
                    True if (self.starting_epoch != 0 or i > 0) else False
                ),
            )  # save statistics to stats file.

            # load_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv') # How to load a csv file if you need to

            out_string = "_".join(
                [
                    "{}_{:.4f}".format(key, np.mean(value))
                    for key, value in current_epoch_losses.items()
                ]
            )
            # create a string to use to report our epoch metrics
            epoch_elapsed_time = (time.time() - epoch_start_time)  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}:".format(epoch_idx),
                out_string,
                "epoch time",
                epoch_elapsed_time,
                "seconds",
            )
            self.state["model_epoch"] = epoch_idx

            # Save current model and override the previous "latest" model
            for model_idx in [epoch_idx, "latest"]:
                self.save_model(
                    model_save_dir=self.experiment_saved_models,
                    model_save_name="train_model",
                    model_idx=model_idx,
                    lowest_val_loss_model_idx=self.lowest_val_loss_model_idx,
                    lowest_val_loss_model_value=self.lowest_val_loss_model_value,
                )

            ################################################################
            ##### Plot Gradient Flow at each Epoch during Training  ######
            print("Generating Gradient Flow Plot at epoch {}".format(epoch_idx))
            plt = self.plot_grad_flow(self.model.named_parameters())

            ### Adding a colorbar to the plot to show the epochs
            if ruonceflag==True:
                numepochs = self.num_epochs
                colormap = matplotlib.cm.get_cmap("viridis") 
                norm = Normalize(vmin=1, vmax=numepochs)
                sm = ScalarMappable(cmap=colormap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=plt.gca(), aspect=30, pad=0.02)
                cbar.set_label("Epochs")
                cbar.set_ticks([1, numepochs])
                cbar.set_ticklabels(["Epoch 1", f"Epoch {numepochs}"])
                ruonceflag=False

            if not os.path.exists(os.path.join(self.experiment_saved_models, "gradient_flow_plots")):
                os.mkdir(os.path.join(self.experiment_saved_models, "gradient_flow_plots"))
            print("save_loc: ",os.path.join(self.experiment_saved_models,"gradient_flow_plots","epoch{}.pdf".format(str(epoch_idx)),),)
            plt.savefig(os.path.join(self.experiment_saved_models,"gradient_flow_plots","epoch{}.pdf".format(str(epoch_idx)),))
            ################################################################

        print("Generating test set evaluation metrics")
        self.load_model( # TODO: USEFUL? if not, ERRASE
            model_save_dir=self.experiment_saved_models,
            model_idx=self.lowest_val_loss_model_idx,
            # load best validation model
            model_save_name="train_model",
        )
        current_epoch_losses = {}
        current_epoch_losses["test_loss"] = []
        current_epoch_losses.update({f"test_{metric_name}": [] for metric_name in self.metrics.keys()})
        
        
        with tqdm.tqdm(total=len(self.test_data)) as pbar_test:
            for points, targets in self.test_data:  # sample batch
                loss, performance_metrics = self.run_iter(points, targets, train=False) 
                current_epoch_losses["test_loss"].append(loss)
                for metric_name, metric_value in performance_metrics.items():
                    current_epoch_losses[f"test_{metric_name}"].append(metric_value)
                pbar_test.update(1)
                pbar_test.set_postfix(loss=f"{loss:.4f}")

        test_losses = {
            key: [np.mean(value)] for key, value in current_epoch_losses.items()
        }  # save test set metrics in dict format
        save_statistics(
            experiment_log_dir=self.experiment_logs,
            filename="test_summary.csv",
            # save test set metrics on disk in .csv format
            stats_dict=test_losses,
            current_epoch=0,
            continue_from_mode=False,
        )

        return total_losses, test_losses
