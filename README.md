# QuantBox

AI-powered quantitative analysis agent with secure sandbox execution.

## Architecture

```
┌─────────────┐     gRPC      ┌─────────────────┐
│  Main API   │ ────────────► │  Sandbox Worker │
│  (FastAPI)  │               │  (Python Exec)  │
└─────────────┘               └─────────────────┘
       │
       │ Kimi API
       ▼
┌─────────────┐
│  Moonshot   │
│    LLM      │
└─────────────┘
```

## Features

- **Market Data**: Fetch OHLCV data via yfinance
- **Correlation Analysis**: Calculate asset correlations with heatmap
- **Backtesting**: Simple moving average strategy backtests
- **Prediction Models**: Linear regression price predictions
- **Chart Generation**: Matplotlib charts returned as Base64

## Quick Start

```bash
# Setup
cp .env.example .env
# Edit .env with your KIMI_API_KEY

# Build & Run
make build
make up

# Open browser
open http://localhost:8000
```

## Project Structure

```
├── apps/
│   ├── main-api/          # FastAPI + Kimi integration
│   └── sandbox-worker/    # gRPC server for code execution
├── protos/
│   └── sandbox.proto      # gRPC protocol definition
└── docker-compose.yaml
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `KIMI_API_KEY` | Moonshot API key |
| `KIMI_BASE_URL` | API endpoint (default: https://api.moonshot.ai/v1) |
| `SANDBOX_HOST` | Sandbox gRPC address (default: localhost:50051) |

## License

MIT
