"""Bias and fairness analysis for the cats/dogs classifier.

Analyzes model performance across proxy subgroups based on image characteristics
(brightness, saturation, color temperature) to detect potential biases.
"""
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple

from src.utils.logging import get_logger
from src.data.split import get_split_dataset

logger = get_logger(__name__)


def compute_image_brightness(images: tf.Tensor) -> np.ndarray:
    """Compute mean brightness (luminance) for each image."""
    # Convert to grayscale using luminance weights
    gray = 0.2989 * images[:, :, :, 0] + 0.5870 * images[:, :, :, 1] + 0.1140 * images[:, :, :, 2]
    return tf.reduce_mean(gray, axis=[1, 2]).numpy()


def compute_image_saturation(images: tf.Tensor) -> np.ndarray:
    """Compute mean saturation for each image using HSV."""
    r, g, b = images[:, :, :, 0], images[:, :, :, 1], images[:, :, :, 2]
    max_c = tf.maximum(tf.maximum(r, g), b)
    min_c = tf.minimum(tf.minimum(r, g), b)
    delta = max_c - min_c

    # Saturation = delta / max (avoid div by 0)
    sat = tf.where(max_c > 0, delta / max_c, 0.0)
    return tf.reduce_mean(sat, axis=[1, 2]).numpy()


def compute_image_color_temp(images: tf.Tensor) -> np.ndarray:
    """Approximate color temperature: warm (reddish) vs cool (bluish)."""
    r_mean = tf.reduce_mean(images[:, :, :, 0], axis=[1, 2])
    b_mean = tf.reduce_mean(images[:, :, :, 2], axis=[1, 2])
    return (r_mean - b_mean).numpy()


def define_subgroups(images: tf.Tensor) -> Dict[str, np.ndarray]:
    """Define proxy subgroups based on image characteristics."""
    brightness = compute_image_brightness(images)
    saturation = compute_image_saturation(images)
    color_temp = compute_image_color_temp(images)

    subgroups = {}

    # Brightness subgroups (low/medium/high)
    b_p33, b_p66 = np.percentile(brightness, [33, 66])
    subgroups["brightness_low"] = brightness <= b_p33
    subgroups["brightness_medium"] = (brightness > b_p33) & (brightness <= b_p66)
    subgroups["brightness_high"] = brightness > b_p66

    # Saturation subgroups
    s_p33, s_p66 = np.percentile(saturation, [33, 66])
    subgroups["saturation_low"] = saturation <= s_p33
    subgroups["saturation_medium"] = (saturation > s_p33) & (saturation <= s_p66)
    subgroups["saturation_high"] = saturation > s_p66

    # Color temperature subgroups
    t_p33, t_p66 = np.percentile(color_temp, [33, 66])
    subgroups["color_temp_cool"] = color_temp <= t_p33
    subgroups["color_temp_neutral"] = (color_temp > t_p33) & (color_temp <= t_p66)
    subgroups["color_temp_warm"] = color_temp > t_p66

    return subgroups


def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_mask: np.ndarray,
    class_names: List[str]
) -> Dict:
    """Compute metrics for a specific subgroup."""
    if subgroup_mask.sum() == 0:
        return {"count": 0, "accuracy": 0.0}

    yt = y_true[subgroup_mask]
    yp = y_pred[subgroup_mask]

    accuracy = float(np.mean(yt == yp))

    per_class = {}
    for i, cls in enumerate(class_names):
        mask = yt == i
        if mask.sum() > 0:
            per_class[cls] = {
                "precision": float(np.mean(yp[mask] == i)),
                "recall": float(np.mean(yt[mask] == i)),
                "support": int(mask.sum()),
            }

    return {
        "count": int(subgroup_mask.sum()),
        "accuracy": accuracy,
        "per_class": per_class,
    }


def run_bias_analysis(
    model_path: str = "models/best_model/model.keras",
    output_dir: str = "reports/analysis",
) -> Dict:
    """Run full bias/fairness analysis."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Load test data
    test_ds = get_split_dataset("test", batch_size=64, shuffle=False)

    # Collect all predictions
    all_images, all_labels, all_preds = [], [], []
    for images, labels in test_ds:
        preds = model(images, training=False)
        all_images.append(images)
        all_labels.append(labels.numpy())
        all_preds.append(np.argmax(preds, axis=1))

    all_images = tf.concat(all_images, axis=0)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    class_names = ["Cat", "Dog"]

    # Define subgroups
    subgroups = define_subgroups(all_images)

    # Compute metrics per subgroup
    results = {"overall": compute_subgroup_metrics(all_labels, all_preds, np.ones(len(all_labels), dtype=bool), class_names)}

    bias_findings = []
    overall_acc = results["overall"]["accuracy"]

    for sg_name, sg_mask in subgroups.items():
        sg_metrics = compute_subgroup_metrics(all_labels, all_preds, sg_mask, class_names)
        results[sg_name] = sg_metrics

        if sg_metrics["count"] > 0:
            acc_diff = abs(sg_metrics["accuracy"] - overall_acc)
            if acc_diff > 0.05:
                bias_findings.append({
                    "subgroup": sg_name,
                    "accuracy": sg_metrics["accuracy"],
                    "accuracy_gap": acc_diff,
                    "count": sg_metrics["count"],
                    "severity": "HIGH" if acc_diff > 0.10 else "MEDIUM",
                })

    # Fairness metrics
    accuracies = [v["accuracy"] for v in results.values() if v["count"] > 0]
    fairness_metrics = {
        "accuracy_range": float(max(accuracies) - min(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "min_accuracy": float(min(accuracies)),
        "max_accuracy": float(max(accuracies)),
        "num_subgroups_analyzed": len(accuracies),
        "bias_findings_count": len(bias_findings),
    }

    output = {
        "results": results,
        "fairness_metrics": fairness_metrics,
        "bias_findings": bias_findings,
    }

    # Save report
    report_path = Path(output_dir) / "bias_fairness_report.json"
    with open(report_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Bias analysis complete. Report saved to {report_path}")
    if bias_findings:
        logger.warning(f"Found {len(bias_findings)} potential bias findings!")
        for finding in bias_findings:
            logger.warning(f"  {finding['subgroup']}: acc={finding['accuracy']:.4f}, gap={finding['accuracy_gap']:.4f}, severity={finding['severity']}")
    else:
        logger.info("No significant bias findings detected.")

    return output


if __name__ == "__main__":
    run_bias_analysis()
