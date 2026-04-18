# Agentic AI Data Science System

A fully agentic ML pipeline powered by **Claude + LangGraph + FastAPI + React**.

## Architecture

```
User uploads CSV
      │
      ▼
[Data Understanding Agent] ──generates──▶ step1_understanding.py
      │
      ▼
[runner.py] ──executes──▶ stdout (structured JSON)
      │ error? loop back (max 10 retries)
      ▼
[Data Analyst Agent] ──generates──▶ step2_analysis.py
      │
      ▼
[runner.py] ──executes──▶ stdout
      │
      ▼
⏸ HUMAN APPROVAL (React UI button)
      │
      ▼
[ML Engineer Agent] ──selects algo + generates──▶ step3_ml.py
      │
      ▼
[runner.py] ──executes──▶ metrics JSON
      │
      ▼
[Evaluation Agent] ──analyzes──▶ verdict (pass/retry)
      │
      ▼
Results Dashboard + model.pkl download
```

## Setup

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
```

Set your Anthropic API key:
```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-...

# Linux/Mac
export ANTHROPIC_API_KEY=sk-ant-...
```

Start the server:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Or double-click `start_backend.bat` on Windows.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Or double-click `start_frontend.bat` on Windows.

Then open **http://localhost:5173** in your browser.

## Usage

1. Upload a CSV dataset
2. Choose task type (Classification / Regression / Clustering)
3. Enter target column (if supervised)
4. Click **Start Agent Pipeline**
5. Watch the **Agent Console** stream live events
6. After data analysis, click **Proceed to Modeling** to approve
7. Download `model.pkl` when complete

## Project Structure

```
backend/
├── agents/
│   ├── data_understanding.py   # Step 1 agent
│   ├── data_analyst.py         # Step 2 agent
│   ├── ml_engineer.py          # Step 3 agent
│   ├── evaluator.py            # Step 4 agent
├── tools/
│   └── runner.py               # Executes generated .py files
├── graph/
│   └── agent_graph.py          # LangGraph pipeline
├── api/
│   └── main.py                 # FastAPI endpoints + SSE
├── generated_code/             # step1/2/3 .py files written here
├── outputs/                    # plots, model.pkl, evaluation.json
└── uploads/                    # uploaded datasets

frontend/
└── src/
    ├── App.jsx
    └── components/
        ├── UploadScreen.jsx        # File upload + config
        ├── PipelineDashboard.jsx   # Main view (SSE consumer)
        ├── AgentConsole.jsx        # Live event stream
        ├── CodeViewer.jsx          # Generated code tabs
        └── ResultsDashboard.jsx    # Metrics + download
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/upload` | Upload CSV dataset |
| POST | `/start` | Start pipeline |
| GET | `/stream/{id}` | SSE event stream |
| GET | `/state/{id}` | Current state snapshot |
| POST | `/approve/{id}` | Human approval to continue |
| GET | `/code/{id}/{step}` | Get generated code |
| GET | `/download/{id}/model` | Download model.pkl |
