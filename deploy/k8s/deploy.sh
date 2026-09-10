#!/bin/bash
# Скрипт для деплоя в Kubernetes кластер

set -e

echo "==================================="
echo "Kubernetes Deployment"
echo "==================================="
echo ""

# Проверка наличия kubectl
if ! command -v kubectl &> /dev/null; then
    echo "kubectl not found. Please install kubectl first."
    exit 1
fi

# Проверка наличия kind (optional)
if command -v kind &> /dev/null; then
    echo "Checking kind cluster..."
    if ! kind get clusters 2>/dev/null | grep -q "gradient-descent"; then
        echo "Creating kind cluster..."
        kind create cluster --name gradient-descent
    fi
    echo "Using kind cluster: gradient-descent"
    kubectl cluster-info --context kind-gradient-descent
fi

echo ""
echo "Building Docker images..."
docker build -t gradient-descent-coordinator:latest -f coordinator/Dockerfile .
docker build -t gradient-descent-worker:latest -f worker/Dockerfile .
docker build -t gradient-descent-viewer:latest -f viewer/Dockerfile .

if command -v kind &> /dev/null; then
    echo ""
    echo "Loading images to kind cluster..."
    kind load docker-image gradient-descent-coordinator:latest --name gradient-descent
    kind load docker-image gradient-descent-worker:latest --name gradient-descent
    kind load docker-image gradient-descent-viewer:latest --name gradient-descent
fi

echo ""
echo "Applying Kubernetes manifests..."
kubectl apply -f deploy/k8s/base/namespace.yaml
kubectl apply -f deploy/k8s/base/
kubectl apply -f deploy/k8s/networking/
kubectl apply -f deploy/k8s/resilience/
kubectl apply -f deploy/k8s/autoscaling/

echo ""
echo "Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=coordinator -n gradient-descent --timeout=60s
kubectl wait --for=condition=ready pod -l app=worker -n gradient-descent --timeout=60s

echo ""
echo "==================================="
echo "Deployment complete!"
echo "==================================="
echo ""
echo "Check status:"
echo "  kubectl get pods -n gradient-descent"
echo ""
echo "View logs:"
echo "  kubectl logs -f deployment/coordinator -n gradient-descent"
echo "  kubectl logs -f deployment/worker -n gradient-descent"
echo ""
echo "Access viewer:"
if command -v kind &> /dev/null; then
    echo "  kubectl port-forward -n gradient-descent service/viewer 8080:80"
    echo "  Then open http://localhost:8080"
else
    echo "  http://<node-ip>:30080"
fi
echo ""
echo "Scale workers:"
echo "  kubectl scale deployment worker --replicas=5 -n gradient-descent"
echo ""
