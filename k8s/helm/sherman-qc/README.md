# Sherman QC Helm Chart

AI-Powered Quality Control System for CNC Manufacturing

## Overview

This Helm chart deploys the Sherman QC system on Kubernetes with support for:
- Backend API (FastAPI)
- Frontend UI (React/Nginx)
- Worker processes (Celery)
- PostgreSQL database
- Redis cache
- Ingress with TLS
- Horizontal Pod Autoscaling
- Prometheus monitoring

## Prerequisites

- Kubernetes 1.23+
- Helm 3.8+
- PV provisioner with ReadWriteMany support (for shared storage)
- cert-manager (optional, for TLS)
- NGINX Ingress Controller (optional)
- Prometheus Operator (optional, for monitoring)

## Installation

### Quick Start (Development)

```bash
# Add chart dependencies
helm dependency update ./k8s/helm/sherman-qc

# Install with development values
helm install sherman-qc ./k8s/helm/sherman-qc \
  -f ./k8s/helm/sherman-qc/values-dev.yaml \
  --set postgresql.auth.password=dev-password
```

### Production Deployment

```bash
# Create namespace
kubectl create namespace sherman-qc

# Create secrets first
kubectl create secret generic sherman-qc-ai-secrets \
  --from-literal=anthropic-api-key=YOUR_API_KEY \
  -n sherman-qc

kubectl create secret generic sherman-qc-auth-secrets \
  --from-literal=jwt-secret=$(openssl rand -hex 32) \
  -n sherman-qc

# Install with production values
helm install sherman-qc ./k8s/helm/sherman-qc \
  -f ./k8s/helm/sherman-qc/values-prod.yaml \
  --set postgresql.auth.password=SECURE_PASSWORD \
  --set redis.auth.password=SECURE_PASSWORD \
  -n sherman-qc
```

## Configuration

### Key Values

| Parameter | Description | Default |
|-----------|-------------|---------|
| `backend.replicaCount` | Backend replicas | `2` |
| `backend.autoscaling.enabled` | Enable HPA for backend | `true` |
| `frontend.replicaCount` | Frontend replicas | `2` |
| `worker.enabled` | Enable async workers | `true` |
| `ingress.enabled` | Enable Ingress | `true` |
| `ingress.hosts[0].host` | Ingress hostname | `qc.example.com` |
| `postgresql.enabled` | Deploy PostgreSQL | `true` |
| `redis.enabled` | Deploy Redis | `true` |
| `ai.provider` | AI provider (claude/gemini/openai) | `claude` |
| `monitoring.enabled` | Enable Prometheus monitoring | `true` |

### Environment-Specific Values

The chart includes pre-configured values files:

- `values-dev.yaml` - Development (minimal resources, no HA)
- `values-staging.yaml` - Staging (moderate resources, basic HA)
- `values-prod.yaml` - Production (full resources, HA, security)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Ingress                                │
│                  (nginx-ingress-controller)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌─────────────────┐            ┌─────────────────┐
│    Frontend     │            │    Backend      │
│    (Nginx)      │───────────▶│    (FastAPI)    │
│   React SPA     │            │                 │
└─────────────────┘            └────────┬────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
           ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
           │ PostgreSQL  │     │    Redis    │     │   Workers   │
           │  (Primary)  │     │   (Cache)   │     │  (Celery)   │
           └──────┬──────┘     └─────────────┘     └─────────────┘
                  │
                  ▼
           ┌─────────────┐
           │  Replicas   │
           └─────────────┘
```

## Storage

The chart creates two PersistentVolumeClaims:

- **scans** - Uploaded scan files (STL, PLY, OBJ, STEP, IGES)
- **reports** - Generated reports and heatmaps

For production, use a storage class that supports ReadWriteMany:
- AWS: EFS
- GCP: Filestore
- Azure: Azure Files

## Monitoring

When `monitoring.enabled=true`, the chart creates:

- **ServiceMonitor** - Prometheus scrape configuration
- **PrometheusRules** - Alert rules for:
  - High error rate (>5%)
  - High latency (p95 > 2s)
  - Pod not ready
  - Database connection exhaustion

## Upgrading

```bash
# Update dependencies
helm dependency update ./k8s/helm/sherman-qc

# Upgrade release
helm upgrade sherman-qc ./k8s/helm/sherman-qc \
  -f ./k8s/helm/sherman-qc/values-prod.yaml \
  -n sherman-qc
```

## Uninstallation

```bash
helm uninstall sherman-qc -n sherman-qc

# Note: PVCs are not deleted automatically
kubectl delete pvc -l app.kubernetes.io/name=sherman-qc -n sherman-qc
```

## Troubleshooting

### Check pod status
```bash
kubectl get pods -l app.kubernetes.io/name=sherman-qc -n sherman-qc
```

### View logs
```bash
kubectl logs -f -l app.kubernetes.io/component=backend -n sherman-qc
```

### Debug deployment
```bash
helm template sherman-qc ./k8s/helm/sherman-qc -f values-dev.yaml --debug
```

## Support

- Documentation: https://github.com/82labs/sherman-qc
- Issues: https://github.com/82labs/sherman-qc/issues
