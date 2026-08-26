"""Data drift detection using statistical tests.

Compares training vs test distribution using:
- Population Stability Index (PSI)
- Kolmogorov-Smirnov test
- Wasserstein distance
"""
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from scipy import stats
from typing import Dict

from src.utils.logging import get_logger
from src.data.split import get_split_dataset

logger = get_logger(__name__)


def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 20) -> float:
    """Compute Population Stability Index (PSI).

    PSI < 0.1: no significant drift
    PSI 0.1-0.2: moderate drift
    PSI > 0.2: significant drift
    """
    eps = 1e-4
    breakpoints = np.linspace(
        min(reference.min(), current.min()) - eps,
        max(reference.max(), current.max()) + eps,
        n_bins + 1,
    )

    ref_hist, _ = np.histogram(reference, bins=breakpoints)
    cur_hist, _ = np.histogram(current, bins=breakpoints)

    ref_pct = ref_hist / ref_hist.sum() + eps
    cur_pct = cur_hist / cur_hist.sum() + eps

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def ks_test(reference: np.ndarray, current: np.ndarray) -> Dict:
    """Two-sample Kolmogorov-Smirnov test."""
    stat, p_value = stats.ks_2samp(reference, current)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < 0.05),
    }


def wasserstein_distance(reference: np.ndarray, current: np.ndarray) -> float:
    """Compute Wasserstein distance between distributions."""
    return float(stats.wasserstein_distance(reference, current))


def extract_image_features(images: tf.Tensor) -> Dict[str, np.ndarray]:
    """Extract summary statistics from images."""
    features = {}

    # Per-channel statistics
    for ch, name in enumerate(["red", "green", "blue"]):
        channel = images[:, :, :, ch].numpy()
        features[f"{name}_mean"] = np.mean(channel, axis=(1, 2))
        features[f"{name}_std"] = np.std(channel, axis=(1, 2))

    # Overall statistics
    gray = tf.reduce_mean(images, axis=-1).numpy()
    features["brightness"] = np.mean(gray, axis=(1, 2))
    features["contrast"] = np.std(gray, axis=(1, 2))

    # Spatial frequency (edge detection approximation)
    dx = gray[:, :, 1:] - gray[:, :, :-1]
    dy = gray[:, 1:, :] - gray[:, :-1, :]
    features["edge_density"] = np.mean(np.abs(dx), axis=(1, 2)) + np.mean(np.abs(dy), axis=(1, 2))

    return features


def run_drift_analysis(
    output_dir: str = "reports/analysis",
    max_samples: int = 1000,
) -> Dict:
    """Run full data drift analysis comparing train vs test distributions."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load datasets (limited samples for speed)
    train_ds = get_split_dataset("train", batch_size=64, shuffle=False)
    test_ds = get_split_dataset("test", batch_size=64, shuffle=False)

    # Collect images (limited)
    train_images, test_images = [], []
    for images, _ in train_ds:
        train_images.append(images)
        if sum(len(b) for b in train_images) >= max_samples:
            break
    for images, _ in test_ds:
        test_images.append(images)
        if sum(len(b) for b in test_images) >= max_samples:
            break

    train_images = tf.concat(train_images, axis=0)[:max_samples]
    test_images = tf.concat(test_images, axis=0)[:max_samples]

    logger.info(f"Loaded {len(train_images)} train, {len(test_images)} test images")

    # Extract features
    train_features = extract_image_features(train_images)
    test_features = extract_image_features(test_images)

    # Run drift tests per feature
    drift_results = {}
    drift_findings = []

    for feature_name in train_features:
        ref = train_features[feature_name]
        cur = test_features[feature_name]

        psi = compute_psi(ref, cur)
        ks = ks_test(ref, cur)
        wd = wasserstein_distance(ref, cur)

        drift_results[feature_name] = {
            "psi": float(psi),
            "psi_severity": "significant" if psi > 0.2 else ("moderate" if psi > 0.1 else "none"),
            "ks_test": ks,
            "wasserstein_distance": float(wd),
        }

        if psi > 0.1 or ks["drift_detected"]:
            drift_findings.append({
                "feature": feature_name,
                "psi": float(psi),
                "ks_p_value": float(ks["p_value"]),
                "wasserstein_distance": float(wd),
                "severity": "HIGH" if psi > 0.2 else "MEDIUM",
            })

    # Summary
    n_drifted = len(drift_findings)
    n_total = len(drift_results)
    max_psi = max(r["psi"] for r in drift_results.values())

    summary = {
        "num_features_tested": n_total,
        "num_drifted": n_drifted,
        "drift_percentage": float(n_drifted / n_total * 100),
        "max_psi": max_psi,
        "overall_drift_status": (
            "SIGNIFICANT" if max_psi > 0.2
            else "MODERATE" if max_psi > 0.1
            else "NONE"
        ),
    }

    output = {
        "summary": summary,
        "feature_results": drift_results,
        "drift_findings": drift_findings,
        "train_samples": len(train_images),
        "test_samples": len(test_images),
    }

    # Save report
    report_path = Path(output_dir) / "data_drift_report.json"
    with open(report_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Data drift analysis complete. Report saved to {report_path}")
    logger.info(f"Overall drift status: {summary['overall_drift_status']}")
    if drift_findings:
        logger.warning(f"Found {n_drifted}/{n_total} features with drift")
    else:
        logger.info("No significant drift detected across features.")

    return output


if __name__ == "__main__":
    run_drift_analysis()
