import torch
import torch.distributed as dist

class MetricsEvaluator:
    def __init__(self, metrics_dict, distributed=False, device="cuda"):
        """
        Initializes a metrics evaluator.

        Args:
            metrics_dict (dict): Dictionary of metric functions.
            distributed (bool, optional): If True, enables distributed evaluation. Defaults to False.
            device (torch.device, optional): Device for evaluation. Defaults to None.
        """
        self.metrics_dict = metrics_dict
        self.distributed = distributed
        self.device = device
        self.running_metrics = {k: {} for k in self.metrics_dict.keys()}
        self.batch_counts = {k: 0 for k in self.metrics_dict.keys()}
        self.final_metrics = {k: {} for k in self.metrics_dict.keys()}

    def update_metrics(self, gts, preds):
        """
        Updates the metrics with new ground truth and prediction values.

        Args:
            gts (list): List of ground truth values.
            preds (list): List of predicted values.
        """
        if self.metrics_dict is None:
            return

        with torch.no_grad():
            for task, task_metrics in self.metrics_dict.items():
                for metric_func, metric_kwargs in task_metrics:
                    metric_name = metric_func.__name__
                    batch_metric = metric_func(gts[task], preds[task], **metric_kwargs)

                    if metric_name not in self.running_metrics[task]:
                        self.running_metrics[task][metric_name] = 0.0
                    self.running_metrics[task][metric_name] += batch_metric
                self.batch_counts[task] += 1

    def compute_final_metrics(self):
        """
        Computes the final metrics after accumulating all batches.
        """
        final_metrics = {task: {} for task in self.metrics_dict.keys()}
        for task, metric_values in self.running_metrics.items():
            batch_count = self.batch_counts[task]
            for metric_name, total_value in metric_values.items():
                if self.distributed:
                    total_value = self._average_across_gpus(total_value)
                final_metrics[task][metric_name] = total_value / batch_count

        return final_metrics
    
    def _average_across_gpus(self, tensor_value):
        """Average a scalar tensor across all GPUs."""
        if not self.distributed:
            return tensor_value

        tensor = torch.tensor([tensor_value], dtype=torch.float32, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item() / dist.get_world_size()
    
    def reset(self):
        """
        Resets the running metrics and batch counts.
        """
        self.running_metrics = {k: {} for k in self.metrics_dict.keys()}
        self.batch_counts = {k: 0 for k in self.metrics_dict.keys()}