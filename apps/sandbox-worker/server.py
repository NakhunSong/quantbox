import grpc
from concurrent import futures
import subprocess
import sys
import json
import time
import tempfile
import os
import base64
import glob

import sandbox_pb2
import sandbox_pb2_grpc

VERSION = "0.2.0"


class SandboxServicer(sandbox_pb2_grpc.SandboxServiceServicer):
    def Execute(self, request, context):
        code = request.code
        timeout = request.timeout_seconds or 30

        start_time = time.time()

        # Create temp directory for execution
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_script = os.path.join(temp_dir, "script.py")

            # Wrapper to capture result and handle matplotlib
            wrapped_code = f"""
import json
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend to Agg (non-interactive)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Change to temp directory for saving plots
os.chdir("{temp_dir}")

_result_ = None

{code}

# Save any open figures
if plt.get_fignums():
    plt.savefig("__chart__.png", dpi=100, bbox_inches='tight', facecolor='white')
    plt.close('all')

# Output result
if _result_ is not None:
    try:
        print("__RESULT_JSON__:" + json.dumps(_result_))
    except:
        print("__RESULT_JSON__:" + json.dumps(str(_result_)))
"""
            with open(temp_script, "w") as f:
                f.write(wrapped_code)

            try:
                result = subprocess.run(
                    [sys.executable, temp_script],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=temp_dir,
                )

                logs = result.stdout
                error = result.stderr if result.returncode != 0 else ""

                # Parse result JSON
                result_json = ""
                output_lines = []
                for line in logs.split("\n"):
                    if line.startswith("__RESULT_JSON__:"):
                        result_json = line[16:]
                    else:
                        output_lines.append(line)
                logs = "\n".join(output_lines).strip()

                # Check for chart image
                image_base64 = ""
                chart_path = os.path.join(temp_dir, "__chart__.png")
                if os.path.exists(chart_path):
                    with open(chart_path, "rb") as img_file:
                        image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

                execution_time = time.time() - start_time

                return sandbox_pb2.ExecuteResponse(
                    success=(result.returncode == 0),
                    logs=logs,
                    result_json=result_json,
                    image_base64=image_base64,
                    error=error,
                    execution_time=execution_time,
                )

            except subprocess.TimeoutExpired:
                return sandbox_pb2.ExecuteResponse(
                    success=False,
                    logs="",
                    result_json="",
                    image_base64="",
                    error=f"Execution timed out after {timeout} seconds",
                    execution_time=timeout,
                )
            except Exception as e:
                return sandbox_pb2.ExecuteResponse(
                    success=False,
                    logs="",
                    result_json="",
                    image_base64="",
                    error=str(e),
                    execution_time=time.time() - start_time,
                )

    def HealthCheck(self, request, context):
        return sandbox_pb2.HealthResponse(healthy=True, version=VERSION)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    sandbox_pb2_grpc.add_SandboxServiceServicer_to_server(
        SandboxServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    print(f"Sandbox server started on port 50051 (version {VERSION})")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
