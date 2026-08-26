"""Run all analyses and log results to MLflow."""
import json
import mlflow
from pathlib import Path

from src.utils.config import load_config
from src.utils.logging import get_logger, setup_logging
from src.analysis.bias_fairness import run_bias_analysis
from src.analysis.data_drift import run_drift_analysis
from src.analysis.adversarial import run_robustness_analysis

logger = get_logger(__name__)


def run_all_analyses():
    """Run all analyses and log to MLflow."""
    setup_logging()
    config = load_config()

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    logger.info("Running all analyses...")

    with mlflow.start_run(run_name="analysis-suite") as run:
        # 1. Bias/Fairness Analysis
        logger.info("1/3 Running bias/fairness analysis...")
        bias_results = run_bias_analysis()

        mlflow.log_metrics({
            "bias_accuracy_range": bias_results["fairness_metrics"]["accuracy_range"],
            "bias_accuracy_std": bias_results["fairness_metrics"]["accuracy_std"],
            "bias_findings_count": bias_results["fairness_metrics"]["bias_findings_count"],
        })

        # 2. Data Drift Analysis
        logger.info("2/3 Running data drift analysis...")
        drift_results = run_drift_analysis()

        mlflow.log_metrics({
            "drift_max_psi": drift_results["summary"]["max_psi"],
            "drift_num_features_tested": drift_results["summary"]["num_features_tested"],
            "drift_num_drifted": drift_results["summary"]["num_drifted"],
        })

        # 3. Adversarial Robustness Analysis
        logger.info("3/3 Running adversarial robustness analysis...")
        robustness_results = run_robustness_analysis()

        mlflow.log_metrics({
            "adversarial_clean_accuracy": robustness_results["summary"]["clean_accuracy"],
            "adversarial_worst_case_accuracy": robustness_results["summary"]["worst_case_accuracy"],
            "adversarial_robustness_ratio": robustness_results["summary"]["robustness_ratio"],
            "adversarial_findings_count": robustness_results["summary"]["findings_count"],
        })

        # Log artifacts
        for report_name in [
            "bias_fairness_report.json",
            "data_drift_report.json",
            "adversarial_robustness_report.json",
        ]:
            report_path = Path("reports/analysis") / report_name
            if report_path.exists():
                mlflow.log_artifact(str(report_path))

        # Summary
        summary = {
            "bias_status": "PASS" if bias_results["fairness_metrics"]["bias_findings_count"] == 0 else "WARN",
            "drift_status": drift_results["summary"]["overall_drift_status"],
            "robustness_status": robustness_results["summary"]["robustness_verdict"],
        }

        mlflow.log_params({
            "analysis_bias_status": summary["bias_status"],
            "analysis_drift_status": summary["drift_status"],
            "analysis_robustness_status": summary["robustness_status"],
        })

        logger.info(f"Analysis complete. Run ID: {run.info.run_id}")
        logger.info(f"Summary: {json.dumps(summary, indent=2)}")

        return summary


if __name__ == "__main__":
    run_all_analyses()
