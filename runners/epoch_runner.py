import torch
import torch.distributed as dist
import tqdm
import numpy as np


class EpochRunner:
    def __init__(
        self,
        model,
        device,
        optimizer=None,
        loss_func=None,
        scheduler_handler=None,
        metrics_evaluator=None,
        use_amp=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.metrics_evaluator = metrics_evaluator
        self.scheduler_handler = scheduler_handler
        self.device = device
        self.use_amp = use_amp
        self.amp_scaler = torch.amp.GradScaler(enabled=use_amp)

        # Get rank for distributed training
        self.is_distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0

    def run_epoch(self, mode, epoch_num, dataloader):
        mode = mode.lower()
        if mode == "train":
            return self.training_epoch(epoch_num, dataloader)
        elif mode == "validate":
            return self.validation_epoch(epoch_num, dataloader)
        elif mode == "test":
            return self.test_epoch(epoch_num, dataloader)
        elif mode == "inference":
            return self.inference_epoch(epoch_num, dataloader)
        else:
            raise ValueError(
                f"Unsupported mode: {mode}. Use 'train', 'validate', or 'test'."
            )

    def to_cpu(self, tensor_dict):
        return {
            key: tensor.detach().cpu() for key, tensor in tensor_dict.items()
        }

    def average_across_gpus(self, tensor_value):
        """Average a scalar tensor across all GPUs."""
        if not self.is_distributed:
            return tensor_value.item()

        tensor = torch.tensor([tensor_value], dtype=torch.float32, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item() / dist.get_world_size()

    def training_epoch(self, epoch_num, dataloader):
        self.model.train()
        running_loss = 0

        mode = "Training"
        tqdm_desc = f"[Epoch {epoch_num + 1} {mode:>10}]"

        with torch.enable_grad():
            for batch in tqdm.tqdm(
                dataloader, desc=tqdm_desc, disable=(self.rank != 0)
            ):
                inputs = {
                    key: item.to(self.device, non_blocking=True)
                    for key, item in batch["inputs"].items()
                }
                labels = {
                    key: item.to(self.device, non_blocking=True)
                    for key, item in batch["labels"].items()
                }

                with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                    batch_logits = self.model(**inputs)
                    loss = self.loss_func(batch_logits, labels)

                self.optimizer.zero_grad()
                self.amp_scaler.scale(loss).backward()
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()

                # Synchronize loss across all GPUs
                loss_value = self.average_across_gpus(loss)
                running_loss += loss_value

                if self.metrics_evaluator:
                    self.metrics_evaluator.update_metrics(
                        labels, batch_logits
                    )

            if self.scheduler_handler is not None:
                self.scheduler_handler.step()

            if self.metrics_evaluator:
                metrics_dict = self.metrics_evaluator.compute_final_metrics()
                self.metrics_evaluator.reset()
            else:
                metrics_dict = {}
            epoch_loss = running_loss / len(dataloader)

            return metrics_dict, epoch_loss

    def validation_epoch(
        self,
        epoch_num,
        dataloader,
    ):
        self.model.eval()
        running_loss = 0

        mode_name = "Validating"
        tqdm_desc = f"[Epoch {epoch_num + 1} {mode_name:>10}]"

        with torch.no_grad():
            for batch in tqdm.tqdm(
                dataloader, desc=tqdm_desc, disable=(self.rank != 0)
            ):
                inputs = {
                    key: item.to(self.device, non_blocking=True)
                    for key, item in batch["inputs"].items()
                }
                labels = {
                    key: item.to(self.device, non_blocking=True)
                    for key, item in batch["labels"].items()
                }

                with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                    batch_logits = self.model(**inputs)
                    loss = self.loss_func(batch_logits, labels)

                loss_value = self.average_across_gpus(loss)
                running_loss += loss_value

                if self.metrics_evaluator:
                    self.metrics_evaluator.update_metrics(
                        labels, batch_logits
                    )

            if self.metrics_evaluator:
                metrics_dict = self.metrics_evaluator.compute_final_metrics()
                self.metrics_evaluator.reset()
            else:
                metrics_dict = {}
            epoch_loss = running_loss / len(dataloader)

            return metrics_dict, epoch_loss

    def test_epoch(self, epoch_num, dataloader):
        self.model.eval()
        all_feats, all_gts, all_logits = {}, {}, {}

        mode_name = "Testing"
        tqdm_desc = f"[Epoch {epoch_num + 1} {mode_name:>10}]"

        with torch.no_grad():
            for batch in tqdm.tqdm(
                dataloader, desc=tqdm_desc, disable=(self.rank != 0)
            ):
                inputs = {
                    key: item.to(self.device, non_blocking=True)
                    for key, item in batch["inputs"].items()
                }
                labels = {
                    key: item.to(self.device, non_blocking=True)
                    for key, item in batch["labels"].items()
                }

                with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                    batch_logits = self.model(**inputs)

                labels = self.to_cpu(labels)
                batch_logits = self.to_cpu(batch_logits)

                for task in batch_logits.keys():
                    all_logits.setdefault(task, []).extend(batch_logits[task])
                    all_gts.setdefault(task, []).extend(labels[task])

                if hasattr(self.model, "extract_features"):
                    batch_feats = self.model.extract_features(**inputs)
                    all_feats.setdefault("features", []).extend(
                        self.to_cpu(batch_feats)
                    )

            return all_gts, all_logits, all_feats

    def inference_epoch(self, epoch_num, dataloader):
        self.model.eval()
        all_feats, all_logits = {}, {}
        mode = "Inference"
        tqdm_desc = f"[Epoch {epoch_num + 1} {mode:>10}]"
        with torch.no_grad():
            for batch in tqdm.tqdm(
                dataloader, desc=tqdm_desc, disable=(self.rank != 0)
            ):
                inputs = {
                    key: item.to(self.device, non_blocking=True)
                    for key, item in batch["inputs"].items()
                }

                with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                    batch_logits = self.model(**inputs)

                for task, logits in batch_logits.items():
                    all_logits.setdefault(task, []).extend(self.to_cpu(logits))

                if hasattr(self.model, "extract_features"):
                    batch_feats = self.model.extract_features(**inputs)
                    all_feats.setdefault("features", []).extend(
                        self.to_cpu(batch_feats)
                    )

            return all_logits, all_feats