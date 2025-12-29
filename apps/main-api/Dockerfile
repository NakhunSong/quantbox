FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY apps/main-api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy proto and generate gRPC code
COPY protos/sandbox.proto ./protos/
RUN python -m grpc_tools.protoc -I./protos \
    --python_out=. \
    --grpc_python_out=. \
    ./protos/sandbox.proto

# Copy application code
COPY apps/main-api/main.py .
COPY apps/main-api/kimi_tools.py .
COPY apps/main-api/static ./static

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
