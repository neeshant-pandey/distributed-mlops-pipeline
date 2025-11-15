#!/bin/bash
# Deployment script for DeepLOB MLOps pipeline

set -e

echo "Deploying DeepLOB MLOps Pipeline..."

# Build and start services
docker-compose -f docker/docker-compose.yml up -d --build

echo "Waiting for services to be ready..."
sleep 10

# Check service health
echo "Checking service health..."

# Check API
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✓ API is healthy"
else
    echo "✗ API health check failed"
fi

# Check MLflow
if curl -s http://localhost:5000 > /dev/null; then
    echo "✓ MLflow is running"
else
    echo "✗ MLflow check failed"
fi

# Check Prometheus
if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo "✓ Prometheus is healthy"
else
    echo "✗ Prometheus check failed"
fi

# Check Grafana
if curl -s http://localhost:3000/api/health > /dev/null; then
    echo "✓ Grafana is healthy"
else
    echo "✗ Grafana check failed"
fi

echo ""
echo "Deployment complete!"
echo ""
echo "Access the services at:"
echo "  API:        http://localhost:8000"
echo "  API Docs:   http://localhost:8000/docs"
echo "  MLflow:     http://localhost:5000"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana:    http://localhost:3000 (admin/admin)"
echo ""
