.PHONY: build up down logs proto clean

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
