@echo off
title Ransomware Detection System

echo ====================================
echo RANSOMWARE DETECTION SYSTEM STARTUP
echo ====================================
echo.

echo Checking model files...
if not exist "best_pso_ransomware_model.h5" (
    echo ERROR: best_pso_ransomware_model.h5 not found!
    echo Please copy your model files to this directory.
    pause
    exit /b 1
)

if not exist "pso_ransomware_model_xgboost_20250811_154446.pkl" (
    echo ERROR: XGBoost model file not found!
    echo Please copy your model files to this directory.
    pause
    exit /b 1
)

echo ✓ Model files found
echo.

echo Starting Ransomware Detection System...
echo Dashboard will be available at: http://localhost:8000/dashboard
echo API documentation at: http://localhost:8000/docs
echo.

echo Press Ctrl+C to stop the system
echo.

python main.py --host 127.0.0.1 --port 8000

echo.
echo System stopped.
pause
