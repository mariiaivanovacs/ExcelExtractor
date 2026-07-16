# Quick Start - Django Table OCR

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
cd webapp
pip install -r requirements.txt
```

Expected packages:
- Django 5.0+
- channels 4.0+
- daphne 4.0+

### Step 2: Start Server
```bash
# Option A - Script
./start_server.sh          # Mac/Linux
start_server.bat           # Windows

# Option B - Direct
python manage.py runserver
```

### Step 3: Use the App
Open: **http://localhost:8000**

1. Upload image (drag & drop or browse)
2. Watch real-time progress
3. Download CSV + visualization

---

## 📁 What's Different from FastAPI Version?

### Django Advantages:
✅ **No async/event loop issues** - Channels handles it properly
✅ **Simpler codebase** - Django's conventions reduce code
✅ **Better static files** - Built-in static file handling
✅ **Production ready** - Proven deployment patterns
✅ **Extensible** - Easy to add features later

### File Structure:
```
webapp/
├── manage.py              # Django management
├── config/                # Project settings
│   ├── settings.py       # Configuration
│   ├── asgi.py           # ASGI + WebSocket
│   └── urls.py           # URL routing
└── tableocr/              # Main app
    ├── views.py          # HTTP endpoints
    ├── consumers.py      # WebSocket handler
    ├── pipeline_runner.py # OCR pipeline
    ├── templates/        # HTML
    └── static/           # CSS + JS
```

---

## 🔧 Common Commands

### Run Server
```bash
python manage.py runserver          # Default: localhost:8000
python manage.py runserver 8001     # Different port
python manage.py runserver 0.0.0.0:8000  # All interfaces
```

### Check Configuration
```bash
python manage.py check              # Validate settings
```

### Collect Static Files (Production)
```bash
python manage.py collectstatic
```

---

## 🐛 Troubleshooting

### "Module not found: django"
```bash
pip install -r requirements.txt
```

### Port Already in Use
```bash
# Mac/Linux
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### WebSocket Not Connecting
1. Check browser console (F12)
2. Ensure server is running
3. Check URL: `ws://localhost:8000/ws/process/{session-id}/`

### Pipeline Fails
Check terminal output for specific error. Most common:
- Missing dependencies in parent project
- Model files not found
- Invalid image format

---

## 📊 How It Works

### Upload Flow:
1. **Frontend** → POST `/upload/` → **Django View**
2. View saves file, returns session ID
3. Frontend opens WebSocket: `/ws/process/{session-id}/`
4. WebSocket consumer starts pipeline in background
5. Progress updates sent via WebSocket
6. Results copied to media folder
7. Frontend loads previews and enables downloads

### Pipeline Steps (14 total):
```
7%   - Preprocessing
14%  - Remove borders
21%  - Clean cells
28%  - Extract blobs
35%  - Clean blobs
42%  - Extract words
50%  - Resize words
57%  - Improve quality
64%  - Segment characters
71%  - Sort CSV
78%  - Remove invalid chars
85%  - Sort again
92%  - Predict digits
100% - Combine results
```

---

## 🎯 Pro Tips

1. **Test with small images first** - Processing can take 5-30 minutes
2. **Watch terminal output** - Detailed error messages appear there
3. **Keep terminal open** - Close it = server stops
4. **Use Chrome/Firefox** - Best WebSocket support
5. **Check browser console** - F12 for debugging

---

## 🔄 Terminal Interface Still Works!

The original CLI interface is unchanged:
```bash
cd ..  # Go to project root
python run_pipeline.py data/input/your_image.jpg
```

Both interfaces share the same pipeline code.

---

## 📚 Need More Info?

- [README.md](README.md) - Full documentation
- Check terminal for errors
- Browser console (F12) for frontend issues
- Test pipeline directly: `python tableocr/pipeline_runner.py`

---

## ✨ Features

- ✅ Real-time progress (WebSocket)
- ✅ Modern UI (no frameworks needed)
- ✅ Drag & drop upload
- ✅ CSV table preview
- ✅ Image visualization
- ✅ Auto file cleanup (1 hour)
- ✅ Works alongside terminal

---

That's it! Just install, run, and start processing tables. 🎉
