@echo off
REM Start Django server for Table OCR (Windows)

echo 🚀 Starting Table Extraction OCR (Django)...
echo.

REM Check if manage.py exists
if not exist "manage.py" (
    echo ❌ Error: manage.py not found. Run from webapp directory.
    pause
    exit /b 1
)

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python not installed
    pause
    exit /b 1
)

REM Check dependencies
echo 📦 Checking dependencies...
python -c "import django" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Django not found. Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo ✅ Starting Django development server...
echo 📍 Application: http://localhost:8000
echo.
echo Press Ctrl+C to stop
echo.

REM Start server
python manage.py runserver 0.0.0.0:8000
