import os

import matplotlib.pyplot as plt
from IPython import get_ipython
from IPython.display import display


class TrainingPlotter:
    def __init__(self, metrics_names, save_path=None):
        self.metrics_names = metrics_names
        self.num_metrics = len(metrics_names)
        self.train_metrics = {name: [] for name in metrics_names}
        self.val_metrics = {name: [] for name in metrics_names}
        self.train_losses = []
        self.val_losses = []
        self.save_path = save_path  # Directory to save plots if running in a script

        self.in_notebook = self._is_notebook()
        if not self.in_notebook:
            plt.ioff()  # Enable interactive mode in scripts

        self.fig, (*metric_axes, self.loss_ax) = plt.subplots(
            ncols=self.num_metrics + 1, nrows=1
        )
        self.metric_axes = dict(zip(metrics_names, metric_axes))
        self.fig.set_size_inches(5 * (self.num_metrics + 1), 4)

        self.metric_lines = {}
        for metric_name in metrics_names:
            ax = self.metric_axes[metric_name]
            self.metric_lines[metric_name] = self._init_ax(
                ax, xlabel="Epoch", ylabel=metric_name, ylim=(0, 1.0)
            )

        self.loss_train_line, self.loss_val_line = self._init_ax(
            self.loss_ax, xlabel="Epoch", ylabel="Loss", ylim=(0, 3.0)
        )

        for name in self.metrics_names:
            set_axis_spines(self.metric_axes[name])
        set_axis_spines(self.loss_ax)
        self.fig.tight_layout()

        self.loss_initialized = False

        if self.in_notebook:
            self.hfig = display(self.fig, display_id=True)

    def _is_notebook(self):
        """Check if running inside a Jupyter Notebook."""
        try:
            return get_ipython() is not None
        except ImportError:
            return False

    def _init_ax(self, ax, **kwargs):
        (train_line,) = ax.plot(
            [], [], label="Train", marker="o", color="blue", linewidth=1, markersize=4
        )
        (val_line,) = ax.plot(
            [],
            [],
            label="Validation",
            marker="o",
            color="red",
            linewidth=1,
            markersize=4,
        )
        ax.legend()
        ax.yaxis.grid(True)
        ax.set_xticks([0])
        ax.set(**kwargs)
        return train_line, val_line

    def _update_ax(self, ax, train_line, val_line, train_data, val_data, epoch):
        epochs = list(range(1, epoch + 2))
        train_line.set_data(epochs, train_data)
        val_line.set_data(epochs, val_data)

        ax.set_xlim(0, epoch + 2)
        ax.set_xticks(range(0, epoch + 2, max(1, (epoch + 2) // 10)))

    def _update_loss_ylim(self):
        all_losses = self.train_losses + self.val_losses
        max_loss = max(all_losses)

        if not self.loss_initialized or max_loss > self.loss_ax.get_ylim()[1]:
            self.loss_ax.set_ylim(0, max_loss * 1.1)
            self.loss_initialized = True

    def update(self, epoch, train_metrics, val_metrics, train_loss, val_loss):
        for name in self.metrics_names:
            self.train_metrics[name].append(train_metrics[name])
            self.val_metrics[name].append(val_metrics[name])
            train_line, val_line = self.metric_lines[name]
            self._update_ax(
                self.metric_axes[name],
                train_line,
                val_line,
                self.train_metrics[name],
                self.val_metrics[name],
                epoch,
            )

        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self._update_ax(
            self.loss_ax,
            self.loss_train_line,
            self.loss_val_line,
            self.train_losses,
            self.val_losses,
            epoch,
        )

        self._update_loss_ylim()

        if self.in_notebook:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.hfig.update(self.fig)
        else:
            if self.save_path:
                self.save_plot(os.path.join(self.save_path, "training_plot.png"))

    def save_plot(self, path):
        """Saves the plot to the specified path"""
        self.fig.savefig(path)

    def close(self):
        """ "Closes the plot, preventing it from blocking the notebook"""
        if self.in_notebook:
            plt.close(self.fig)  # Prevents memory leaks in Jupyter Notebook


def set_axis_spines(ax, left=True, bottom=True, right=False, top=False):
    """Function to remove axis from a matplotlib axis"""
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.spines["right"].set_visible(right)
    ax.spines["top"].set_visible(top)

