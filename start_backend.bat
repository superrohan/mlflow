@echo off
echo Starting Agentic AI Data Science Backend...
cd /d "%~dp0backend"
set OPENAI_API_KEY=%OPENAI_API_KEY%
python run_server.py
