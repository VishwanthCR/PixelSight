import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import json

from backend.app.main import app

client = TestClient(app)


def test_01_classification_classes_endpoint():
    """Validates /api/v1/classification/classes returns canonical ESA WorldCover 7-class + ignore."""
    response = client.get("/api/v1/classification/classes")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 8  # 7 classes + 1 ignore
    class_names = [c["name"] for c in data]
    assert "Tree" in class_names
    assert "Shrubland" in class_names
    assert "Grassland" in class_names
    assert "Cropland" in class_names
    assert "Built-up" in class_names
    assert "Bare" in class_names
    assert "Water" in class_names
    assert "Ignore" in class_names


def test_02_benchmark_classification_endpoint_structure():
    """Validates /api/v1/results/benchmark/classification matches Section 21 schema."""
    response = client.get("/api/v1/results/benchmark/classification")
    assert response.status_code == 200
    data = response.json()

    # Core keys
    assert "job_id" in data
    assert "model" in data
    assert "evaluation" in data
    assert "overall" in data
    assert "per_class" in data
    assert "distribution" in data
    assert "confusion_matrix" in data
    assert "artifacts" in data
    assert "limitations" in data
    assert "interpretations" in data


def test_03_benchmark_overall_metrics_and_deltas():
    """Validates benchmark overall metrics match verified Plan 4 matched evaluation."""
    response = client.get("/api/v1/results/benchmark/classification")
    assert response.status_code == 200
    overall = response.json()["overall"]

    native = overall["native"]
    sr = overall["sr"]
    delta = overall["delta"]

    # Native values from Plan 4 matched resolution evaluation
    assert native["miou"] == pytest.approx(0.3012, abs=1e-3)
    assert native["accuracy"] == pytest.approx(0.7598, abs=1e-3)
    assert native["dice"] == pytest.approx(0.3961, abs=1e-3)

    # SR values
    assert sr["miou"] == pytest.approx(0.2311, abs=1e-3)
    assert sr["dice"] == pytest.approx(0.3039, abs=1e-3)

    # Delta sr - native
    assert delta["miou"] == pytest.approx(-0.0701, abs=1e-3)
    assert delta["dice"] == pytest.approx(-0.0922, abs=1e-3)


def test_04_per_class_metrics():
    """Validates per-class metrics contain IoU, F1, precision, recall, support, and deltas."""
    response = client.get("/api/v1/results/benchmark/classification")
    assert response.status_code == 200
    per_class = response.json()["per_class"]

    assert len(per_class) == 7
    tree_class = next(c for c in per_class if c["name"] == "Tree")
    assert tree_class["id"] == 0
    assert tree_class["native"]["iou"] is not None
    assert tree_class["sr"]["iou"] is not None
    assert tree_class["delta"]["iou"] is not None
    assert tree_class["native"]["support"] is not None

    builtup_class = next(c for c in per_class if c["name"] == "Built-up")
    assert builtup_class["id"] == 4
    assert builtup_class["native"]["iou"] is not None
    assert builtup_class["sr"]["iou"] is not None


def test_05_confusion_matrix_structure():
    """Validates confusion matrix object contains 7x7 rows for native and SR."""
    response = client.get("/api/v1/results/benchmark/classification")
    assert response.status_code == 200
    conf = response.json()["confusion_matrix"]
    assert "native" in conf
    assert "sr" in conf
    assert len(conf["native"]) == 7
    assert len(conf["sr"]) == 7
    assert len(conf["native"][0]) == 7
    assert len(conf["sr"][0]) == 7


def test_06_class_distribution():
    """Validates class distribution has pixel_count, percent, and area_ha."""
    response = client.get("/api/v1/results/benchmark/classification")
    assert response.status_code == 200
    dist = response.json()["distribution"]
    assert "native" in dist
    assert "sr" in dist
    assert "Tree" in dist["native"]
    assert "percent" in dist["native"]["Tree"]
    assert "area_ha" in dist["native"]["Tree"]


def test_07_no_reference_missing_labels():
    """Validates that a job without reference labels has None for accuracy/mIoU and prediction comparison."""
    # ps_337db1d0 is a real crop job without segmentation labels
    response = client.get("/api/v1/results/ps_337db1d0/classification")
    assert response.status_code == 200
    data = response.json()
    assert data["evaluation"]["has_reference_labels"] is False
    assert data["overall"]["native"]["accuracy"] is None
    assert data["overall"]["native"]["miou"] is None
    assert data["overall"]["sr"]["accuracy"] is None
    assert data["overall"]["sr"]["miou"] is None


def test_08_alias_jobs_classification_route():
    """Validates /api/v1/jobs/{job_id}/classification functions as an alias."""
    response = client.get("/api/v1/jobs/benchmark/classification")
    assert response.status_code == 200
    assert response.json()["job_id"] == "benchmark"


def test_09_csv_export_endpoint():
    """Validates /api/v1/results/{job_id}/classification/statistics.csv returns CSV format."""
    response = client.get("/api/v1/results/benchmark/classification/statistics.csv")
    assert response.status_code == 200
    assert "text/csv" in response.headers.get("content-type", "")
    lines = response.text.strip().split("\n")
    assert len(lines) >= 8  # header + 7 classes
    assert "class_id,class_name" in lines[0]


def test_10_confusion_matrix_json_endpoint():
    """Validates /api/v1/results/{job_id}/classification/confusion_matrix.json."""
    response = client.get("/api/v1/results/benchmark/classification/confusion_matrix.json")
    assert response.status_code == 200
    data = response.json()
    assert "confusion_matrix" in data
    assert "classes" in data
