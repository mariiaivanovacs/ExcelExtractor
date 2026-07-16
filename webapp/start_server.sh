#!/bin/bash
# Start Django server for Table OCR

echo "🚀 Starting Table Extraction OCR (Django)..."
echo ""

# Check if manage.py exists
if [ ! -f "manage.py" ]; then
    echo "❌ Error: manage.py not found. Run from webapp directory."
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 not installed"
    exit 1
fi

# Check dependencies
echo "📦 Checking dependencies..."
python3 -c "import django" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Django not found. Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "✅ Starting Django development server..."
echo "📍 Application: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start server
python3 manage.py runserver 0.0.0.0:8000
