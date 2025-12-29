FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY apps/sandbox-worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy proto and generate gRPC code
COPY protos/sandbox.proto ./protos/
RUN python -m grpc_tools.protoc -I./protos \
    --python_out=. \
    --grpc_python_out=. \
    ./protos/sandbox.proto

# Copy server code
COPY apps/sandbox-worker/server.py .

EXPOSE 50051

CMD ["python", "server.py"]
