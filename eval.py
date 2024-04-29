import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, roc_curve, auc

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def evaluate_segmentation(gt_mask, pred_mask):
    # Ensure masks are binary
    gt_mask_binary = (gt_mask > 127).astype(np.uint8)
    pred_mask_binary = (pred_mask > 127).astype(np.uint8)

    # Flatten the masks to compute the scores
    gt_mask_flat = gt_mask_binary.flatten()
    pred_mask_flat = pred_mask_binary.flatten()

    # Jaccard score, also known as the IoU score
    iou = jaccard_score(gt_mask_flat, pred_mask_flat, pos_label=1)
    dice = f1_score(gt_mask_flat, pred_mask_flat, pos_label=1)
    precision = precision_score(gt_mask_flat, pred_mask_flat, pos_label=1)
    recall = recall_score(gt_mask_flat, pred_mask_flat, pos_label=1)
    accuracy = accuracy_score(gt_mask_flat, pred_mask_flat)
    conf_matrix = confusion_matrix(gt_mask_flat, pred_mask_flat, labels=[0, 1])

    return {
        "IoU": iou,
        "Dice": dice,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "Confusion Matrix": conf_matrix
    }

def process_evaluation(gt_folder, pred_folder):
    metrics_results = {}
    for filename in os.listdir(gt_folder):
        base_name = filename.split('_mask')[0]
        pred_filename = f"{base_name}_07_cleaned_distance.jpg"
        gt_path = os.path.join(gt_folder, filename)
        pred_path = os.path.join(pred_folder, pred_filename)
        
        if os.path.exists(pred_path):
            gt_mask = load_image(gt_path)
            pred_mask = load_image(pred_path)  # This should be your predicted probabilities map
            metrics_results[base_name] = evaluate_segmentation(gt_mask, pred_mask)
        else:
            print(f"Prediction for {base_name} not found.")
    return metrics_results

def visualize_metrics(results, gt_folder, pred_folder):
    # Metrics to lists for plotting
    ious = [metrics['IoU'] for metrics in results.values()]
    dices = [metrics['Dice'] for metrics in results.values()]
    precisions = [metrics['Precision'] for metrics in results.values()]
    recalls = [metrics['Recall'] for metrics in results.values()]
    accuracies = [metrics['Accuracy'] for metrics in results.values()]

    tprs = []  # True positive rates
    aucs = []  # Area under the curve values
    mean_fpr = np.linspace(0, 1, 100)  # Mean false positive rates

    # Initialize figure
    plt.figure(figsize=(18, 10))
    sns.set_style("whitegrid")

    # Plot histogram distributions
    metrics_list = [ious, dices, precisions, recalls, accuracies]
    titles = ['IoU Scores', 'Dice Scores', 'Precision Scores', 'Recall Scores', 'Accuracy Scores']
    for i, metrics in enumerate(metrics_list, 1):
        plt.subplot(2, 3, i)
        sns.histplot(metrics, kde=True)
        plt.title(titles[i-1])

    # ROC Curve subplot
    plt.subplot(2, 3, 6)
    for base_name, metrics in results.items():
        gt_path = os.path.join(gt_folder, f"{base_name}_mask.png")
        pred_path = os.path.join(pred_folder, f"{base_name}_07_cleaned_distance.jpg")

        gt_mask = load_image(gt_path)
        pred_probs = load_image(pred_path)  # This should be your predicted probabilities map

        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve((gt_mask.flatten() > 127).astype(int), pred_probs.flatten() / 255)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0  # Ensure the curve starts at 0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    # Plot the average ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure the curve ends at 1
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2, alpha=0.8)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', label='Chance', alpha=.8     )
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1-Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()


# Define paths to your ground truth and predicted masks
gt_folder = 'data/gt_img'
pred_folder = 'data/segment'

# Evaluate and print the metrics for each image
results = process_evaluation(gt_folder, pred_folder)

# Visualize the collected metrics and ROC curve
visualize_metrics(results, gt_folder, pred_folder)
