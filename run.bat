@echo off
REM Batch script to run the RL Scheduling Simulator on Windows

echo ========================================
echo  RL Scheduling Simulator
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Python found!
echo.

REM Check if requirements are installed
echo Checking dependencies...
python -c "import numpy, pandas, matplotlib, seaborn, yaml" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Some required packages are missing.
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Error: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo Dependencies OK!
echo.

REM Ask user what to run
echo What would you like to run?
echo   1. Quick Demo
echo   2. Full Simulation
echo   3. Exit
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Running Quick Demo...
    echo.
    python demo.py
) else if "%choice%"=="2" (
    echo.
    echo Running Full Simulation...
    echo.
    cd src
    python main.py
    cd ..
) else if "%choice%"=="3" (
    echo.
    echo Exiting...
    exit /b 0
) else (
    echo.
    echo Invalid choice. Running Quick Demo by default...
    echo.
    python demo.py
)

echo.
echo ========================================
echo  Execution Complete
echo ========================================
pause
