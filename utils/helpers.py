def unravel_metric_dict(metrics):
    """
    Unravels the nested dictionary of metrics into a flat dictionary.
    By default, the metrics dictionary is nested with the task name as the first level key,
    and the metric name as the second level key.
    This function will flatten the dictionary by concatenating the task name with the metric name.

    e.g.
    {
        "task1": {
            "metric1": value1,
            "metric2": value2
        },
        "task2": {
            "metric1": value3,
        }
    }   
    will be converted to
    {
        "task1_metric1": value1,
        "task1_metric2": value2,
        "task2_metric1": value3,
    }
    """
    unraveled_dict = {}
    for task, metrics in metrics.items():
        for metric_name, metric_values in metrics.items():
            unraveled_dict[f"{task}_{metric_name}"] = metric_values
    return unraveled_dict