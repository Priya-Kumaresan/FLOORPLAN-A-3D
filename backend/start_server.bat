@echo off
echo Activating virtual environment...
call ..\sop\Scripts\activate.bat
echo.
echo Starting FastAPI server on http://127.0.0.1:8000
echo Press Ctrl+C to stop
echo.
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000

