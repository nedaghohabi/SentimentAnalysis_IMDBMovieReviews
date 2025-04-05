import logging
from logging.handlers import RotatingFileHandler


class Logger:
    def __init__(
        self,
        name,
        log_file,
        console_output=True,
        console_level=logging.DEBUG,
        file_level=logging.INFO,
        max_bytes=5 * 1024 * 1024,
        backup_count=5,
    ):
        """
        Logger with different log levels, including file handler and optional console output.

        Args:
            name (str): The name of the logger.
            log_file (str): The file path for the log file.
            console_output (bool): Whether to log to the console. Default is True.
            console_level (int): The logging level for the console. Default is logging.DEBUG.
            file_level (int): The logging level for the log file. Default is logging.INFO.
            max_bytes (int): Max size in bytes for the rotating log file. Default is 5 MB.
            backup_count (int): Number of backup log files to keep. Default is 5.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        self.file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        self.file_handler.setLevel(file_level)
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)

        if console_output:
            self.console_handler = logging.StreamHandler()
            self.console_handler.setLevel(console_level)
            self.console_handler.setFormatter(formatter)
            self.logger.addHandler(self.console_handler)

    def log_message(self, message, level=logging.INFO):
        self.logger.log(level, message)

    def metrics_to_string(self, metrics_dict):
        """
        Convert a dictionary of metrics to a formatted string.

        Args:
            metrics_dict (dict): Dictionary containing metric names and values.

        Returns:
            str: Formatted string of metrics.
        """
        return ", ".join(
            [
                f"{name}: {value:.4f}"
                for name, value in metrics_dict.items()
            ]
        )

    def log_training(self, epoch, loss, metrics_dict, lr=0.0001):
        metrics_str = self.metrics_to_string(metrics_dict)
        self.logger.info(
            f"[Epoch: {epoch:03d}]      Train info: Loss: {loss:.4f}, {metrics_str}, LR: {lr:.6f}"
        )

    def log_validation(self, epoch, loss, metrics_dict):
        metrics_str = self.metrics_to_string(metrics_dict)
        self.logger.info(
            f"[Epoch: {epoch:03d}] Validation info: Loss: {loss:.4f}, {metrics_str}"
        )

    def log_test(self, epoch, loss, metrics_dict):
        metrics_str = self.metrics_to_string(metrics_dict)
        self.logger.info(
            f"[Epoch: {epoch:03d}]       Test info: Loss: {loss:.4f}, {metrics_str}"
        )

    def log_exception(self, e):
        self.logger.error(f"Exception occurred: {str(e)}", exc_info=True)

    def close(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()