.PHONY: help all install install-dev dvc-init dvc-pull dvc-push dvc-repro train test lint build run clean argocd-install argocd-deploy argocd-status monitoring-up monitoring-down monitoring-k8s

# Full setup: install, test, build, deploy
all: install-dev test build
	@echo "Setup complete. Run 'make deploy-kind' or 'make argocd-deploy' to deploy."

# Default target
help:
	@echo "Available targets:"
	@echo "  all            - Full setup: install, test, build"
	@echo "  install        - Install production dependencies"
	@echo "  install-dev    - Install development dependencies"
	@echo "  dvc-init       - Initialize DVC"
	@echo "  dvc-pull       - Pull data/models from DVC remote"
	@echo "  dvc-push       - Push data/models to DVC remote"
	@echo "  dvc-repro      - Reproduce DVC pipeline"
	@echo "  train          - Train model (run DVC train stage)"
	@echo "  test           - Run tests"
	@echo "  lint           - Run linters"
	@echo "  build          - Build Docker image"
	@echo "  run            - Run API locally"
	@echo "  mlflow-ui      - Start MLflow UI"
	@echo "  argocd-install - Install Argo CD on cluster"
	@echo "  argocd-deploy  - Deploy app via Argo CD"
	@echo "  argocd-status  - Show Argo CD app status"
	@echo "  deploy-kind    - Deploy to Kind (kubectl)"
	@echo "  smoke-test     - Run smoke tests"
	@echo "  monitoring-up  - Start monitoring stack (Prometheus + Grafana)"
	@echo "  monitoring-down - Stop monitoring stack"
	@echo "  monitoring-k8s - Deploy monitoring to Kind cluster"
	@echo "  clean          - Clean up generated files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

dvc-init:
	dvc init
	dvc remote add -d localremote /tmp/dvc-store

dvc-pull:
	dvc pull

dvc-push:
	dvc push

dvc-repro:
	dvc repro

train: dvc-repro

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	black --check src/ tests/
	mypy src/

build:
	docker build -f docker/Dockerfile -t cats-dogs-api:latest .

run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow_artifacts --host 0.0.0.0 --port 5000

mlflow-up:
	kubectl apply -f k8s/mlflow/mlflow.yaml
	@echo "MLflow UI: http://localhost:30500"

mlflow-down:
	kubectl delete -f k8s/mlflow/mlflow.yaml

# Argo CD targets
argocd-install:
	bash scripts/setup-argocd.sh

argocd-deploy:
	kubectl apply -f argocd/project.yaml
	kubectl apply -f argocd/application.yaml

argocd-status:
	argocd app get cats-dogs-api-production || echo "Argo CD CLI not available or app not found"
	kubectl get applications -n argocd || echo "No Argo CD applications found"

argocd-sync:
	argocd app sync cats-dogs-api-production

argocd-delete:
	argocd app delete cats-dogs-api-production --cascade

# Kubernetes deployment targets
deploy-kind:
	kind create cluster --name mlops-cluster 2>/dev/null || true
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml
	kubectl rollout status deployment/cats-dogs-api -n mlops-cats-dogs --timeout=300s

smoke-test:
	bash scripts/smoke_test.sh

# Monitoring targets
monitoring-up:
	docker-compose --profile monitoring up -d
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000 (admin/admin)"

monitoring-down:
	docker-compose --profile monitoring down

monitoring-k8s:
	kubectl apply -f k8s/monitoring/namespace.yaml
	kubectl apply -f k8s/monitoring/
	@echo "Waiting for monitoring pods..."
	kubectl wait --for=condition=ready pod -l app=prometheus -n monitoring --timeout=120s
	kubectl wait --for=condition=ready pod -l app=grafana -n monitoring --timeout=120s
	@echo "Prometheus: kubectl port-forward -n monitoring svc/prometheus 9090:9090"
	@echo "Grafana: kubectl port-forward -n monitoring svc/grafana 3000:3000"

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache
	rm -rf mlflow.db mlflow_artifacts mlruns
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
