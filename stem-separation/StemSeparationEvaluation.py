# Imports and Setup
import pandas as pd
from collections import defaultdict
import numpy as np
import nussl

from StemSeparationDeployment import process_audio

METRICS = ["SI-SDR", "SI-SIR", "SI-SAR", "SD-SDR", "SNR", "SRR", "SI-SDRi", "SD-SDRi", "SNRi", "MIX-SI-SDR", "MIX-SD-SDR", "MIX-SNR"]

def get_scores(model, item):
    vocals_estimate_audio = process_audio(item, model)
    estimates = {
        "vocals": vocals_estimate_audio,
        "accompaniment": item["mix"] - vocals_estimate_audio,
    }

    sources = {
        "vocals": item["sources"]["vocals"],
        "accompaniment": item["sources"]["accompaniment"],
    }
    source_keys = list(sources.keys())

    estimates = [estimates[k] for k in source_keys]
    sources = [sources[k] for k in source_keys]

    evaluator = nussl.evaluation.BSSEvalScale(
        sources, estimates, source_labels=source_keys
    )
    scores = evaluator.evaluate()

    return scores

def evaluateModel(model, data, num_items, eval_path):
    vocals_metrics, accompaniment_metrics = defaultdict(list), defaultdict(list)

    for i in range(num_items):
        item = data[i]
        scores = get_scores(model, item)

        for metric in METRICS:
            vocals_metrics[metric].extend(scores["vocals"][metric])
            accompaniment_metrics[metric].extend(scores["accompaniment"][metric])
    
    df = pd.DataFrame(columns=["Metric", "Vocals Mean", "Vocals Standard Deviation", "Accompaniment Mean", "Accompaniment Standard Deviation"])
    for metric in METRICS:
        metric_row = {
            "Metric": metric,
            "Vocals Mean": np.mean(vocals_metrics[metric]),
            "Vocals Standard Deviation": np.std(vocals_metrics[metric]),
            "Accompaniment Mean": np.mean(accompaniment_metrics[metric]),
            "Accompaniment Standard Deviation": np.std(accompaniment_metrics[metric]),
        }
        df = pd.concat([pd.DataFrame([metric_row], columns=df.columns), df], ignore_index=True)

    
    df.to_csv(eval_path, index=False)
    
    return df
