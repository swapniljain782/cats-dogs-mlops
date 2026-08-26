"""Adversarial robustness testing.

Tests model resilience against:
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)
- Random noise perturbations
"""
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List

from src.utils.logging import get_logger
from src.data.split import get_split_dataset

logger = get_logger(__name__)


def fgsm_attack(
    model: tf.keras.Model,
    images: tf.Tensor,
    labels: tf.Tensor,
    epsilon: float = 0.01,
) -> tf.Tensor:
    """Fast Gradient Sign Method attack."""
    images = tf.cast(images, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(images)
        predictions = model(images, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

    gradients = tape.gradient(loss, images)
    signed_grad = tf.sign(gradients)
    adversarial_images = images + epsilon * signed_grad
    adversarial_images = tf.clip_by_value(adversarial_images, 0.0, 1.0)
    return adversarial_images


def pgd_attack(
    model: tf.keras.Model,
    images: tf.Tensor,
    labels: tf.Tensor,
    epsilon: float = 0.01,
    alpha: float = 0.001,
    num_steps: int = 10,
) -> tf.Tensor:
    """Projected Gradient Descent attack (iterative FGSM)."""
    images = tf.cast(images, tf.float32)
    adversarial = tf.identity(images)

    for _ in range(num_steps):
        with tf.GradientTape() as tape:
            tape.watch(adversarial)
            predictions = model(adversarial, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

        gradients = tape.gradient(loss, adversarial)
        signed_grad = tf.sign(gradients)
        adversarial = adversarial + alpha * signed_grad

        # Project back into epsilon-ball
        delta = adversarial - images
        delta = tf.clip_by_value(delta, -epsilon, epsilon)
        adversarial = images + delta
        adversarial = tf.clip_by_value(adversarial, 0.0, 1.0)

    return adversarial


def random_noise_attack(
    images: tf.Tensor,
    epsilon: float = 0.01,
) -> tf.Tensor:
    """Random uniform noise attack."""
    noise = tf.random.uniform(images.shape, -epsilon, epsilon)
    adversarial = tf.clip_by_value(images + noise, 0.0, 1.0)
    return adversarial


def evaluate_robustness(
    model: tf.keras.Model,
    images: tf.Tensor,
    labels: tf.Tensor,
    attack_fn,
    attack_name: str,
    epsilon: float,
) -> Dict:
    """Evaluate model accuracy under a specific attack."""
    adversarial_images = attack_fn(model, images, labels, epsilon)

    # Accuracy on adversarial examples
    preds_adv = model(adversarial_images, training=False)
    adv_acc = float(tf.reduce_mean(tf.cast(tf.argmax(preds_adv, axis=1) == labels, tf.float32)))

    # Confidence analysis
    confidences = tf.reduce_max(preds_adv, axis=1).numpy()
    mean_confidence = float(np.mean(confidences))

    # How many predictions changed
    preds_clean = model(images, training=False)
    clean_preds = tf.argmax(preds_clean, axis=1).numpy()
    adv_preds = tf.argmax(preds_adv, axis=1).numpy()
    flip_rate = float(np.mean(clean_preds != adv_preds))

    return {
        "attack": attack_name,
        "epsilon": epsilon,
        "accuracy": adv_acc,
        "mean_confidence": mean_confidence,
        "prediction_flip_rate": flip_rate,
    }


def run_robustness_analysis(
    model_path: str = "models/best_model/model.keras",
    output_dir: str = "reports/analysis",
    max_samples: int = 100,
) -> Dict:
    """Run full adversarial robustness analysis."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Load test data
    test_ds = get_split_dataset("test", batch_size=32, shuffle=False)

    all_images, all_labels = [], []
    for images, labels in test_ds:
        all_images.append(images)
        all_labels.append(labels.numpy())
        if sum(len(b) for b in all_images) >= max_samples:
            break

    images = tf.concat(all_images, axis=0)[:max_samples]
    labels = np.concatenate(all_labels)[:max_samples]
    labels = tf.constant(labels)

    logger.info(f"Testing robustness on {len(images)} samples")

    # Baseline accuracy (no attack)
    preds_clean = model(images, training=False)
    clean_acc = float(tf.reduce_mean(tf.cast(tf.argmax(preds_clean, axis=1) == labels, tf.float32)))
    logger.info(f"Clean accuracy: {clean_acc:.4f}")

    # Test across epsilon values
    epsilons = [0.01, 0.1]
    attacks = {
        "fgsm": lambda m, i, l, e: fgsm_attack(m, i, l, e),
        "pgd": lambda m, i, l, e: pgd_attack(m, i, l, e, alpha=e/4, num_steps=5),
        "random_noise": lambda m, i, l, e: random_noise_attack(i, e),
    }

    results = {}
    robustness_findings = []

    for attack_name, attack_fn in attacks.items():
        attack_results = []
        for eps in epsilons:
            metrics = evaluate_robustness(model, images, labels, attack_fn, attack_name, eps)
            attack_results.append(metrics)
            logger.info(f"  {attack_name} eps={eps}: acc={metrics['accuracy']:.4f}, flip_rate={metrics['prediction_flip_rate']:.4f}")

        results[attack_name] = attack_results

        # Check if model fails badly at low epsilon
        low_eps = attack_results[0]
        if low_eps["accuracy"] < 0.5:
            robustness_findings.append({
                "attack": attack_name,
                "epsilon": low_eps["epsilon"],
                "accuracy": low_eps["accuracy"],
                "severity": "HIGH",
                "finding": f"Model accuracy drops below 50% at epsilon={low_eps['epsilon']}",
            })

    # Noise robustness (average over multiple runs)
    noise_accs = []
    for _ in range(3):
        noisy_images = random_noise_attack(images, epsilon=0.05)
        preds_noisy = model(noisy_images, training=False)
        noise_acc = float(tf.reduce_mean(tf.cast(tf.argmax(preds_noisy, axis=1) == labels, tf.float32)))
        noise_accs.append(noise_acc)

    noise_summary = {
        "epsilon": 0.05,
        "mean_accuracy": float(np.mean(noise_accs)),
        "std_accuracy": float(np.std(noise_accs)),
        "num_runs": 3,
    }

    # Overall robustness score
    worst_case_acc = min(
        results[attack][-1]["accuracy"]
        for attack in results
    )
    robustness_score = float(worst_case_acc / clean_acc) if clean_acc > 0 else 0.0

    summary = {
        "clean_accuracy": clean_acc,
        "worst_case_accuracy": worst_case_acc,
        "robustness_ratio": robustness_score,
        "noise_robustness": noise_summary,
        "findings_count": len(robustness_findings),
        "robustness_verdict": (
            "ROBUST" if robustness_score > 0.8
            else "MODERATELY_ROBUST" if robustness_score > 0.5
            else "FRAGILE"
        ),
    }

    output = {
        "summary": summary,
        "attack_results": results,
        "noise_robustness": noise_summary,
        "robustness_findings": robustness_findings,
        "epsilons_tested": epsilons,
        "num_samples_tested": len(images),
    }

    # Save report
    report_path = Path(output_dir) / "adversarial_robustness_report.json"
    with open(report_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Adversarial robustness analysis complete. Report saved to {report_path}")
    logger.info(f"Robustness verdict: {summary['robustness_verdict']} (ratio: {robustness_score:.4f})")

    return output


if __name__ == "__main__":
    run_robustness_analysis()
