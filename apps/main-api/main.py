import os
import json
import grpc
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sandbox_pb2
import sandbox_pb2_grpc
from kimi_tools import QUANT_TOOLS, generate_code_for_tool

app = FastAPI(title="Quant AI Agent API")

# Config
SANDBOX_HOST = os.getenv("SANDBOX_HOST", "localhost:50051")
KIMI_API_KEY = os.getenv("KIMI_API_KEY")
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")

# Kimi client (OpenAI compatible)
client = OpenAI(api_key=KIMI_API_KEY, base_url=KIMI_BASE_URL)

# Conversation history (in-memory, for demo)
conversation_history = []


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    executed_code: str | None = None
    logs: str | None = None
    result_json: str | None = None
    image_base64: str | None = None


def execute_in_sandbox(code: str) -> dict:
    """Execute code in sandbox via gRPC"""
    try:
        logger.info(f"Connecting to sandbox at {SANDBOX_HOST}")
        channel = grpc.insecure_channel(SANDBOX_HOST)
        stub = sandbox_pb2_grpc.SandboxServiceStub(channel)

        request = sandbox_pb2.ExecuteRequest(code=code, timeout_seconds=60)
        response = stub.Execute(request)

        logger.info(f"Sandbox response: success={response.success}, error={response.error[:100] if response.error else None}")

        return {
            "success": response.success,
            "logs": response.logs,
            "result_json": response.result_json,
            "image_base64": response.image_base64,
            "error": response.error,
            "execution_time": response.execution_time,
        }
    except Exception as e:
        logger.error(f"Sandbox execution failed: {e}")
        return {
            "success": False,
            "logs": "",
            "result_json": "",
            "image_base64": "",
            "error": str(e),
            "execution_time": 0,
        }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global conversation_history

    conversation_history.append({"role": "user", "content": request.message})

    executed_code = None
    logs = None
    result_json = None
    image_base64 = None

    # Call Kimi with tools
    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            {
                "role": "system",
                "content": """You are a quantitative analysis AI assistant.
You help users analyze market data, calculate correlations, run backtests, and build predictive models.
When users ask for data analysis, use the available tools to execute Python code in a secure sandbox.
Always explain your analysis results clearly in Korean.""",
            }
        ]
        + conversation_history,
        tools=QUANT_TOOLS,
        tool_choice="auto",
    )

    assistant_message = response.choices[0].message
    logger.info(f"Kimi response: tool_calls={assistant_message.tool_calls}, content={assistant_message.content[:100] if assistant_message.content else None}")

    # Process tool calls if any
    if assistant_message.tool_calls:
        tool_results = []

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_input = json.loads(tool_call.function.arguments)

            # Generate Python code for this tool
            code = generate_code_for_tool(tool_name, tool_input)
            executed_code = code

            # Execute in sandbox
            sandbox_result = execute_in_sandbox(code)
            logs = sandbox_result.get("logs")
            result_json = sandbox_result.get("result_json")
            image_base64 = sandbox_result.get("image_base64")

            tool_results.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": json.dumps(
                        {
                            "success": sandbox_result["success"],
                            "result": sandbox_result["result_json"],
                            "error": sandbox_result["error"],
                            "has_chart": bool(sandbox_result["image_base64"]),
                        }
                    ),
                }
            )

        # Add assistant message with tool calls to history
        conversation_history.append(
            {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_message.tool_calls
                ],
            }
        )

        # Add tool results to history
        for tr in tool_results:
            conversation_history.append(tr)

        # Get final response from Kimi
        final_response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {
                    "role": "system",
                    "content": "You are a quantitative analysis AI assistant. Explain results clearly in Korean.",
                }
            ]
            + conversation_history,
        )

        answer = final_response.choices[0].message.content or ""
        conversation_history.append({"role": "assistant", "content": answer})
    else:
        answer = assistant_message.content or ""
        conversation_history.append({"role": "assistant", "content": answer})

    return ChatResponse(
        answer=answer,
        executed_code=executed_code,
        logs=logs,
        result_json=result_json,
        image_base64=image_base64,
    )


@app.post("/api/reset")
async def reset_conversation():
    global conversation_history
    conversation_history = []
    return {"status": "ok"}


@app.get("/api/health")
async def health():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
