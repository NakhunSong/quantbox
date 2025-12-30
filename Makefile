.PHONY: build up down logs proto clean \
	k8s-setup k8s-build k8s-deploy k8s-status k8s-logs-sandbox k8s-logs-api k8s-url k8s-clean

# Build all images
build:
	docker-compose build

# Start all services
up:
	docker-compose up -d

# Stop all services
down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# Generate proto files locally (for development)
proto:
	python -m grpc_tools.protoc -I./protos \
		--python_out=./apps/sandbox-worker \
		--grpc_python_out=./apps/sandbox-worker \
		./protos/sandbox.proto
	python -m grpc_tools.protoc -I./protos \
		--python_out=./apps/main-api \
		--grpc_python_out=./apps/main-api \
		./protos/sandbox.proto

# Clean up
clean:
	docker-compose down -v --rmi local
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ============================================
# Kubernetes (Minikube) Commands
# ============================================

# Start minikube
k8s-setup:
	minikube start --cpus=2 --memory=4096

# Build images in minikube's docker daemon
k8s-build:
	eval $$(minikube docker-env) && \
	docker build -t quantbox/sandbox-worker:latest -f apps/sandbox-worker/Dockerfile . && \
	docker build -t quantbox/main-api:latest -f apps/main-api/Dockerfile .

# Deploy all resources (reads KIMI_API_KEY from .env)
k8s-deploy:
	@export $$(grep -v '^#' .env | xargs) && \
	kubectl apply -f k8s/namespace.yaml && \
	kubectl apply -f k8s/configmap.yaml && \
	envsubst < k8s/secret.yaml | kubectl apply -f - && \
	kubectl apply -f k8s/sandbox-deployment.yaml && \
	kubectl apply -f k8s/sandbox-service.yaml && \
	kubectl apply -f k8s/main-api-deployment.yaml && \
	kubectl apply -f k8s/main-api-service.yaml

# Check status
k8s-status:
	kubectl get all -n quantbox

# View sandbox logs
k8s-logs-sandbox:
	kubectl logs -f deployment/sandbox -n quantbox

# View main-api logs
k8s-logs-api:
	kubectl logs -f deployment/main-api -n quantbox

# Get service URL
k8s-url:
	minikube service main-api-service -n quantbox --url

# Delete all resources
k8s-clean:
	kubectl delete namespace quantbox
