# Imports and Setup
import pandas as pd
from collections import defaultdict
import numpy as np
import nussl
import matplotlib.pyplot as plt

from StemSeparationDeployment import process_audio

def get_scores(model, item):
    """
    Evaluate our model for a specific audio item and return the evaluation metrics for the vocals and accompaniment

    Args:
    model: the model to evaluate
    item: the specific audio data to evaluate

    Returns:
    scores: a dictionary containing the evaluation metrics for the vocals and accompaniment
    """ 

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
    """
    Evaluate our model's performance with the given array of audio data, formatting and saving the resulting metrics
    to a .csv file on disk.

    Args:
    model: the model to evaluate
    item: the specific audio data to evaluate

    Returns:
    scores: a dictionary containing the evaluation metrics for the vocals and accompaniment
    """ 
    
    METRICS = ["SI-SDR", "SI-SIR", "SI-SAR", "SD-SDR", "SNR", "SRR", "SI-SDRi", "SD-SDRi",
                "SNRi", "MIX-SI-SDR", "MIX-SD-SDR", "MIX-SNR"]
    vocals_metrics, accompaniment_metrics = defaultdict(list), defaultdict(list)

    for i in range(num_items):
        item = data[i]
        scores = get_scores(model, item)

        for metric in METRICS:
            vocals_metrics[metric].extend(scores["vocals"][metric])
            accompaniment_metrics[metric].extend(scores["accompaniment"][metric])
    
    df = pd.DataFrame(columns=["Metric", "Vocals Mean", "Vocals Standard Deviation", 
                               "Accompaniment Mean", "Accompaniment Standard Deviation"])
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

    # Plot Results
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
    axes = axes.flatten()

    metrics_groups = [
            "Vocals Mean", "Vocals Standard Deviation",
            "Accompaniment Mean", "Accompaniment Standard Deviation"]

    colors = plt.cm.get_cmap('tab20', len(df["Metric"]))

    for i, metric in enumerate(metrics_groups):
        axes[i].bar(df["Metric"], df[metric], color=colors(range(len(df["Metric"]))))
        axes[i].set_title(f"{metric} Evaluation")
        axes[i].set_ylabel("Score")
        axes[i].set_xticklabels(df["Metric"], rotation=45, ha="right")

        # Save individual plots
        plt.figure()
        plt.bar(df["Metric"], df[metric], color=colors(range(len(df["Metric"]))))
        plt.title(f"{metric} Evaluation")
        plt.ylabel("Score")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"stem-separation/metrics/{metric.replace(' ', '_').lower()}_evaluation_plot.png")
        plt.close()

    plt.tight_layout()
    plt.savefig("stem-separation/metrics/evaluation_plots.png")
    plt.show()
    
    return df
