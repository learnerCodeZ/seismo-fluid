@echo off
title Fluid Web v6.1
cd /d "%~dp0"
cd /d ".."

set "PORT=8002"
set "URL=http://127.0.0.1:%PORT%"

rem Prefer project venv if present, else python on PATH
set "PYTHON="
if exist "%CD%\.venv\Scripts\python.exe" set "PYTHON=%CD%\.venv\Scripts\python.exe"
if not defined PYTHON if exist "D:\桌面\地质\大创\dataAnalysis\.venv\Scripts\python.exe" set "PYTHON=D:\桌面\地质\大创\dataAnalysis\.venv\Scripts\python.exe"
if not defined PYTHON set "PYTHON=python"

echo ========================================
echo   Earthquake Fluid Visualization
echo   Port %PORT%
echo ========================================
echo.
echo Root: %CD%
echo Python: %PYTHON%
echo.

echo [1/3] Check packages...
"%PYTHON%" -c "import fastapi, uvicorn" 2>nul
if errorlevel 1 (
    "%PYTHON%" -m pip install fastapi uvicorn
)

echo [2/3] Check port...
netstat -ano | findstr ":%PORT% " | findstr "LISTENING" >nul
if not errorlevel 1 (
    echo Port %PORT% already in use. Opening browser...
    echo.
    echo   %URL%/
    echo   %URL%/prediction
    echo   %URL%/anomaly
    echo   %URL%/earthquake
    echo.
    start "" "%URL%/earthquake"
    pause
    exit /b 0
)

echo [3/3] Starting uvicorn on port %PORT% ...
echo.
echo   %URL%/
echo   %URL%/prediction
echo   %URL%/anomaly
echo   %URL%/earthquake
echo.
echo Press Ctrl+C to stop.
echo.

start "" "%URL%/earthquake"
"%PYTHON%" -m uvicorn web.app:app --host 127.0.0.1 --port %PORT%
echo.
echo Stopped.
pause
