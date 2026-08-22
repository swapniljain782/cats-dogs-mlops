#!/usr/bin/env python3
"""Generate Grafana dashboard JSON for MLOps monitoring."""
import json
import sys
from pathlib import Path


def make_stat_panel(id, title, expr, unit="short", thresholds=None, gridPos=None):
    if thresholds is None:
        thresholds = {"mode": "absolute", "steps": [{"color": "green", "value": None}]}
    ds = {"type": "prometheus", "uid": "${DS_PROMETHEUS}"}
    return {
        "datasource": ds,
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "mappings": [],
                "thresholds": thresholds,
                "unit": unit,
            },
            "overrides": [],
        },
        "gridPos": gridPos or {"h": 4, "w": 4, "x": 0, "y": 1},
        "id": id,
        "options": {
            "colorMode": "background",
            "graphMode": "area",
            "justifyMode": "auto",
            "orientation": "auto",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "textMode": "auto",
        },
        "title": title,
        "type": "stat",
        "targets": [{"datasource": ds, "expr": expr, "legendFormat": title, "refId": "A"}],
    }


def make_timeseries(id, title, targets_list, unit="short", gridPos=None, legend_calcs=None):
    ds = {"type": "prometheus", "uid": "${DS_PROMETHEUS}"}
    t_targets = [{"datasource": ds, "expr": t["expr"], "legendFormat": t["legend"], "refId": t.get("ref", "A")} for t in targets_list]
    return {
        "datasource": ds,
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "custom": {
                    "axisCenteredZero": False,
                    "axisColorMode": "text",
                    "axisLabel": "",
                    "axisPlacement": "auto",
                    "barAlignment": 0,
                    "drawStyle": "line",
                    "fillOpacity": 10,
                    "gradientMode": "none",
                    "lineInterpolation": "smooth",
                    "lineWidth": 1,
                    "pointSize": 5,
                    "scaleDistribution": {"type": "linear"},
                    "showPoints": "never",
                    "spanNulls": False,
                    "stacking": {"group": "A", "mode": "none"},
                    "thresholdsStyle": {"mode": "off"},
                },
                "mappings": [],
                "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]},
                "unit": unit,
            },
            "overrides": [],
        },
        "gridPos": gridPos or {"h": 8, "w": 12, "x": 0, "y": 6},
        "id": id,
        "options": {
            "legend": {"calcs": legend_calcs or ["mean", "max"], "displayMode": "table", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
        "title": title,
        "type": "timeseries",
        "targets": t_targets,
    }


def make_pie(id, title, expr, gridPos=None):
    ds = {"type": "prometheus", "uid": "${DS_PROMETHEUS}"}
    return {
        "datasource": ds,
        "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "mappings": [], "unit": "short"}, "overrides": []},
        "gridPos": gridPos or {"h": 8, "w": 6, "x": 18, "y": 6},
        "id": id,
        "options": {"legend": {"displayMode": "table", "placement": "right"}, "pieType": "donut", "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}, "tooltip": {"mode": "single", "sort": "none"}},
        "title": title,
        "type": "piechart",
        "targets": [{"datasource": ds, "expr": expr, "legendFormat": "{{predicted_class}}", "refId": "A"}],
    }


dashboard = {
    "annotations": {"list": [{"builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"}, "enable": True, "hide": True, "iconColor": "rgba(0, 211, 255, 1)", "name": "Annotations & Alerts", "type": "dashboard"}]},
    "description": "MLOps API Monitoring - Cats vs Dogs Classification Service",
    "editable": True,
    "graphTooltip": 1,
    "links": [],
    "panels": [
        # Row: Overview
        {"collapsed": False, "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0}, "id": 100, "title": "Overview", "type": "row"},
        make_stat_panel(1, "Model Status", "model_loaded", thresholds={"mode": "absolute", "steps": [{"color": "red", "value": None}, {"color": "green", "value": 1}]}, gridPos={"h": 4, "w": 4, "x": 0, "y": 1}),
        make_stat_panel(2, "Request Rate (1m)", "sum(rate(http_requests_total[1m]))", unit="reqps", thresholds={"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "yellow", "value": 100}, {"color": "red", "value": 500}]}, gridPos={"h": 4, "w": 4, "x": 4, "y": 1}),
        make_stat_panel(3, "Avg Latency (1m)", "sum(rate(http_request_duration_seconds_sum[1m])) / sum(rate(http_request_duration_seconds_count[1m]))", unit="s", thresholds={"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "yellow", "value": 0.2}, {"color": "red", "value": 0.5}]}, gridPos={"h": 4, "w": 4, "x": 8, "y": 1}),
        make_stat_panel(4, "Error Rate (1m)", "sum(rate(http_requests_total{status=~\"5..\"}[1m])) / sum(rate(http_requests_total[1m]))", unit="percentunit", thresholds={"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "yellow", "value": 0.01}, {"color": "red", "value": 0.05}]}, gridPos={"h": 4, "w": 4, "x": 12, "y": 1}),
        make_stat_panel(5, "Avg Confidence", "sum(rate(model_prediction_confidence_sum[1m])) / sum(rate(model_prediction_confidence_count[1m]))", unit="percentunit", thresholds={"mode": "absolute", "steps": [{"color": "blue", "value": None}, {"color": "green", "value": 0.5}]}, gridPos={"h": 4, "w": 4, "x": 16, "y": 1}),
        make_stat_panel(6, "Total Predictions", "sum(model_predictions_total)", gridPos={"h": 4, "w": 4, "x": 20, "y": 1}),

        # Row: HTTP Traffic
        {"collapsed": False, "gridPos": {"h": 1, "w": 24, "x": 0, "y": 5}, "id": 200, "title": "HTTP Traffic", "type": "row"},
        make_timeseries(10, "Request Rate by Endpoint", [
            {"expr": "sum by (endpoint) (rate(http_requests_total[1m]))", "legend": "{{endpoint}}"},
        ], unit="reqps", gridPos={"h": 8, "w": 12, "x": 0, "y": 6}),
        make_timeseries(11, "Request Rate by Status", [
            {"expr": "sum by (status) (rate(http_requests_total[1m]))", "legend": "Status {{status}}"},
        ], unit="reqps", gridPos={"h": 8, "w": 12, "x": 12, "y": 6}),
        make_timeseries(12, "Response Latency (P50/P95/P99)", [
            {"expr": "histogram_quantile(0.50, sum by (le) (rate(http_request_duration_seconds_bucket[1m])))", "legend": "P50"},
            {"expr": "histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket[1m])))", "legend": "P95"},
            {"expr": "histogram_quantile(0.99, sum by (le) (rate(http_request_duration_seconds_bucket[1m])))", "legend": "P99"},
        ], unit="s", gridPos={"h": 8, "w": 12, "x": 0, "y": 14}),
        make_timeseries(13, "Latency by Endpoint", [
            {"expr": "sum by (endpoint) (rate(http_request_duration_seconds_sum[1m])) / sum by (endpoint) (rate(http_request_duration_seconds_count[1m]))", "legend": "{{endpoint}}"},
        ], unit="s", gridPos={"h": 8, "w": 12, "x": 12, "y": 14}),

        # Row: Model Predictions
        {"collapsed": False, "gridPos": {"h": 1, "w": 24, "x": 0, "y": 22}, "id": 300, "title": "Model Predictions", "type": "row"},
        make_timeseries(20, "Predictions by Class", [
            {"expr": "sum by (predicted_class) (rate(model_predictions_total[1m]))", "legend": "{{predicted_class}}"},
        ], unit="reqps", gridPos={"h": 8, "w": 12, "x": 0, "y": 23}),
        make_pie(21, "Prediction Distribution", "sum by (predicted_class) (model_predictions_total)", gridPos={"h": 8, "w": 6, "x": 12, "y": 23}),
        make_timeseries(22, "Prediction Confidence Over Time", [
            {"expr": "sum by (predicted_class) (rate(model_prediction_confidence_sum[1m])) / sum by (predicted_class) (rate(model_prediction_confidence_count[1m]))", "legend": "{{predicted_class}}"},
        ], unit="percentunit", gridPos={"h": 8, "w": 6, "x": 18, "y": 23}),
        make_timeseries(23, "Prediction Latency Histogram", [
            {"expr": "histogram_quantile(0.50, sum by (le) (rate(model_prediction_latency_seconds_bucket[1m])))", "legend": "P50"},
            {"expr": "histogram_quantile(0.95, sum by (le) (rate(model_prediction_latency_seconds_bucket[1m])))", "legend": "P95"},
        ], unit="s", gridPos={"h": 8, "w": 12, "x": 0, "y": 31}),
        make_timeseries(24, "Prediction Confidence Distribution", [
            {"expr": "histogram_quantile(0.50, sum by (le, predicted_class) (rate(model_prediction_confidence_bucket[1m])))", "legend": "P50 - {{predicted_class}}"},
            {"expr": "histogram_quantile(0.95, sum by (le, predicted_class) (rate(model_prediction_confidence_bucket[1m])))", "legend": "P95 - {{predicted_class}}"},
        ], unit="percentunit", gridPos={"h": 8, "w": 12, "x": 12, "y": 31}),
    ],
    "refresh": "10s",
    "schemaVersion": 38,
    "tags": ["mlops", "api", "monitoring", "cats-dogs"],
    "templating": {"list": []},
    "time": {"from": "now-1h", "to": "now"},
    "timepicker": {},
    "timezone": "browser",
    "title": "MLOps API Dashboard",
    "uid": "mlops-api-dashboard",
    "version": 1,
}


if __name__ == "__main__":
    output = Path(__file__).parent / "mlops-api.json"
    with open(output, "w") as f:
        json.dump(dashboard, f, indent=2)
    print(f"Dashboard written to {output}")
