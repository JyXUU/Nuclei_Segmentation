import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, roc_curve, auc

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def evaluate_segmentation(gt_mask, pred_mask):
    if np.max(gt_mask) == 0 or np.max(pred_mask) == 0:
        return {
            "IoU": 0,
            "Dice": 0,
            "Precision": 0,
            "Recall": 0,
            "Accuracy": 0,
            "Confusion Matrix": [[0, 0], [0, 0]],
        }

    gt_mask_binary = (gt_mask > 127).astype(np.uint8)
    pred_mask_binary = (pred_mask > 127).astype(np.uint8)

    gt_mask_flat = gt_mask_binary.flatten()
    pred_mask_flat = pred_mask_binary.flatten()

    iou = jaccard_score(gt_mask_flat, pred_mask_flat, pos_label=1)
    dice = f1_score(gt_mask_flat, pred_mask_flat, pos_label=1)
    precision = precision_score(gt_mask_flat, pred_mask_flat, pos_label=1, zero_division=1)
    recall = recall_score(gt_mask_flat, pred_mask_flat, pos_label=1, zero_division=1)  
    accuracy = accuracy_score(gt_mask_flat, pred_mask_flat)
    conf_matrix = confusion_matrix(gt_mask_flat, pred_mask_flat, labels=[0, 1])

    return {
        "IoU": iou,
        "Dice": dice,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "Confusion Matrix": conf_matrix,
    }

def process_evaluation(gt_folder, pred_folder):
    metrics_results = {}
    for filename in os.listdir(gt_folder):
        base_name = filename.split('_mask')[0]
        pred_filename = f"{base_name}_06_cleaned_distance.jpg"
        gt_path = os.path.join(gt_folder, filename)
        pred_path = os.path.join(pred_folder, pred_filename)
        
        if os.path.exists(pred_path):
            gt_mask = load_image(gt_path)
            pred_mask = load_image(pred_path)
            metrics_results[base_name] = evaluate_segmentation(gt_mask, pred_mask)
        else:
            print(f"Prediction for {base_name} not found.")
    return metrics_results

def visualize_metrics(results, gt_folder, pred_folder):
    plt.figure(figsize=(18, 12))
    sns.set_style("whitegrid")
    metrics_list = ['IoU', 'Dice', 'Precision', 'Recall', 'Accuracy']

    for i, metric in enumerate(metrics_list, 1):
        scores = [metrics[metric] for metrics in results.values()]
        plt.subplot(3, 3, i)
        sns.histplot(scores, kde=True)
        plt.title(f'{metric} Scores')

    plt.subplot(3, 3, 6)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for base_name, metrics in results.items():
        gt_path = os.path.join(gt_folder, f"{base_name}_mask.png")
        pred_path = os.path.join(pred_folder, f"{base_name}_06_cleaned_distance.jpg")
        gt_mask = load_image(gt_path)
        pred_probs = load_image(pred_path)
        fpr, tpr, thresholds = roc_curve((gt_mask.flatten() > 127).astype(int), pred_probs.flatten() / 255)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2, alpha=0.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', label='Chance', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1-Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

def visualize_metrics_boxplots(results):
    plt.figure(figsize=(8, 6))  # Adjust the figure size to accommodate a single comprehensive plot
    sns.set_style("whitegrid")
    metrics = ['IoU', 'Dice', 'Precision', 'Recall', 'Accuracy']
    data = []

    # Collect all metric scores into a single list with corresponding labels
    for metric in metrics:
        scores = [result[metric] for result in results.values()]
        data.extend([(score, metric) for score in scores])

    # Convert the list into a DataFrame
    import pandas as pd
    df = pd.DataFrame(data, columns=['Score', 'Metric'])

    # Create a boxplot for all metrics
    sns.boxplot(x='Metric', y='Score', data=df)
    plt.title('Distribution of Segmentation Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Score Values')

    plt.tight_layout()
    plt.show()

def calculate_average_scores(results):
    average_scores = {key: np.mean([result[key] for result in results.values()]) for key in ['IoU', 'Dice', 'Precision', 'Recall', 'Accuracy']}
    print("Average Scores:")
    for score_name, score_value in average_scores.items():
        print(f"{score_name}: {score_value:.4f}")

gt_folder = 'data/Gloms_segmented_img'
pred_folder = 'data/Nuclei_segmented'

results = process_evaluation(gt_folder, pred_folder)

visualize_metrics(results, gt_folder, pred_folder)

visualize_metrics_boxplots(results)

calculate_average_scores(results)