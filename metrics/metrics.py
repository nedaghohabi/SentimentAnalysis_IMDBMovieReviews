import torch
import numpy as np


def prepare_inputs(gts, preds, from_logits=True, binary=True):
    if not isinstance(gts, torch.Tensor):
        gts = torch.as_tensor(gts, dtype=torch.long)
    if not isinstance(preds, torch.Tensor):
        preds = torch.as_tensor(preds, dtype=torch.float32)
    if from_logits:
        if binary:
            preds = torch.sigmoid(preds)
        else:
            preds = torch.argmax(preds, dim=-1)
    return gts, preds

def mean_squared_error(gts_list, preds_list):
    gts, preds = prepare_inputs(gts_list, preds_list, False)
    mask = (gts!=-1).all(axis=-1)
    gts = gts[mask]
    preds = preds[mask]
    overall_mse = ((gts - preds) ** 2).sum()
    num_samples = mask.sum()
    return overall_mse / num_samples if num_samples > 0 else 0.0

def accuracy(gts_list, preds_list, from_logits, binary=True, threshold=0.5):
    gts, preds = prepare_inputs(gts_list, preds_list, from_logits=from_logits, binary=binary)
    if binary:
        preds = preds > threshold
    mask = gts!=-1
    gts = gts[mask]
    preds = preds[mask]
    corrects = (preds == gts).sum()
    total = mask.sum()
    return corrects / total if total > 0 else 0.0

def recall(gts_list, preds_list, from_logits, binary=True, threshold=0.5):
    gts, preds = prepare_inputs(gts_list, preds_list, from_logits=from_logits, binary=binary)
    if binary:
        preds = preds > threshold
    mask = gts!=-1
    gts = gts[mask]
    preds = preds[mask]
    corrects = ((preds == 1) & (gts == 1)).sum()
    total = (gts == 1).sum()
    return corrects / total if total > 0 else 0.0

def precision(gts_list, preds_list, from_logits, binary=True, threshold=0.5):
    gts, preds = prepare_inputs(gts_list, preds_list, from_logits=from_logits, binary=binary)
    if binary:
        preds = preds > threshold
    mask = gts!=-1
    gts = gts[mask]
    preds = preds[mask]
    corrects = ((preds == 1) & (gts == 1)).sum()
    total = (preds == 1).sum()
    return corrects / total if total > 0 else 0.0