# Table Extraction OCR - Django Web Application

A modern Django web application for extracting data from table images using advanced OCR technology with real-time WebSocket progress updates.

## Why Django?

This version uses Django instead of FastAPI for better:
- **Threading Support**: Django's ASGI server (Daphne) handles async operations cleanly
- **Built-in Features**: Session management, static files, templates
- **Channels**: Robust WebSocket support via Django Channels
- **Stability**: No event loop closing issues

## Features

- ✅ Modern, responsive UI optimized for desktop
- ✅ Real-time progress updates via WebSockets (Django Channels)
- ✅ Drag & drop file upload
- ✅ Live CSV table preview
- ✅ Image visualization preview
- ✅ Dual download (CSV + PNG)
- ✅ Auto file cleanup (1 hour)
- ✅ Works alongside terminal interface

## Quick Start

### 1. Install Dependencies

```bash
cd webapp
pip install -r requirements.txt
```

### 2. Start Server

**Option A - Using script:**
```bash
# Mac/Linux
./start_server.sh

# Windows
start_server.bat
```

**Option B - Direct:**
```bash
python manage.py runserver 0.0.0.0:8000
```

### 3. Open Browser

Navigate to: **http://localhost:8000**

## Project Structure

```
webapp/
├── manage.py                      # Django management script
├── config/                        # Django project settings
│   ├── settings.py               # Main settings
│   ├── urls.py                   # Root URL config
│   └── asgi.py                   # ASGI config with WebSocket
├── tableocr/                      # Main Django app
│   ├── views.py                  # HTTP request handlers
│   ├── urls.py                   # App URL patterns
│   ├── consumers.py              # WebSocket consumers
│   ├── routing.py                # WebSocket routing
│   ├── pipeline_runner.py        # OCR pipeline wrapper
│   ├── templates/
│   │   └── index.html            # Main page
│   └── static/
│       ├── css/styles.css        # Styling
│       └── js/app.js             # Frontend logic
├── media/                         # Upload/results storage
│   ├── uploads/                  # Temporary uploads
│   └── results/                  # Processed results
└── requirements.txt               # Python dependencies
```

## How It Works

### 1. File Upload
- User uploads image via drag-drop or file browser
- Django view saves file with unique session ID
- Returns session ID to frontend

### 2. WebSocket Connection
- Frontend opens WebSocket to `/ws/process/{session_id}/`
- Sends `start_processing` action
- Django Channels consumer accepts connection

### 3. Processing
- Consumer runs pipeline in thread pool executor
- Progress callbacks send updates via WebSocket
- 14 pipeline steps, each updating percentage
- Frontend displays real-time progress

### 4. Completion
- Results copied to `media/results/`
- Consumer sends `complete` message
- Frontend loads CSV and image previews
- Download buttons activated

### 5. Cleanup
- Files auto-deleted after 1 hour
- Cleanup runs on new uploads
- Results cleared on page refresh

## API Endpoints

### HTTP Endpoints

**Upload Image**
```
POST /upload/
Content-Type: multipart/form-data
Body: file (image)

Response: {"session_id": "uuid", "message": "File uploaded successfully"}
```

**Download CSV**
```
GET /download/csv/{session_id}/
Response: CSV file download
```

**Download Image**
```
GET /download/image/{session_id}/
Response: PNG file download
```

**Preview CSV**
```
GET /preview/csv/{session_id}/
Response: {"rows": 21, "columns": 20, "data": [[...]]}
```

**Preview Image**
```
GET /preview/image/{session_id}/
Response: PNG image
```

### WebSocket Endpoint

**Process Image**
```
WS /ws/process/{session_id}/

Client → Server:
{"action": "start_processing"}

Server → Client:
{"type": "progress", "percentage": 50, "message": "Step description"}
{"type": "complete", "session_id": "uuid"}
{"type": "error", "message": "Error description"}
```

## Configuration

### File Retention
Edit `config/settings.py`:
```python
FILE_RETENTION_HOURS = 1  # Change retention period
```

### Port Configuration
```bash
python manage.py runserver 0.0.0.0:8001  # Use different port
```

### Debug Mode
Edit `config/settings.py`:
```python
DEBUG = False  # Disable for production
```

## Terminal Interface (Still Works!)

The original terminal interface is unchanged:

```bash
cd ..  # Go to project root
python run_pipeline.py data/input/your_image.jpg
```

## Troubleshooting

### "Module not found"
```bash
pip install -r requirements.txt
```

### Port Already in Use
```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
python manage.py runserver 8001
```

### WebSocket Won't Connect
1. Check browser console (F12)
2. Ensure server is running
3. Try different browser
4. Check firewall settings

### Processing Fails
1. Check terminal output for errors
2. Verify main project dependencies installed
3. Test pipeline directly:
```bash
cd tableocr
python pipeline_runner.py
```

### Static Files Not Loading
```bash
python manage.py collectstatic --noinput
```

## Development

### Run with Auto-reload
```bash
python manage.py runserver --noreload  # Disable reload if issues
```

### View Logs
All logs print to terminal. Check for:
- Upload errors
- WebSocket connection issues
- Pipeline step failures

### Test WebSocket
Browser console:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/process/test-id/');
ws.onmessage = (e) => console.log(JSON.parse(e.data));
ws.send(JSON.stringify({action: 'start_processing'}));
```

## Production Deployment

For production:

1. **Set DEBUG = False** in settings.py
2. **Set SECRET_KEY** to random string
3. **Configure ALLOWED_HOSTS**
4. **Use proper database** (optional, not needed for this app)
5. **Run collectstatic**
6. **Use production ASGI server**:

```bash
# Install production server
pip install gunicorn uvicorn

# Run with gunicorn + uvicorn workers
gunicorn config.asgi:application -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

7. **Setup nginx reverse proxy**:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /static/ {
        alias /path/to/webapp/staticfiles/;
    }

    location /media/ {
        alias /path/to/webapp/media/;
    }
}
```

## Dependencies

- **Django 5.0+**: Web framework
- **Channels 4.0+**: WebSocket support
- **Daphne 4.0+**: ASGI server

All compatible with Python 3.13.

## Advantages Over FastAPI Version

✅ **No event loop issues** - Django Channels handles async properly
✅ **Built-in admin** - Can add admin interface if needed
✅ **Better static files** - Django's static file system is robust
✅ **Template system** - Django templates vs manual HTML
✅ **Middleware support** - Easy to add authentication, logging, etc.
✅ **Production ready** - Proven deployment patterns

## License

Same as main project.

## Support

For issues:
1. Check terminal output
2. Check browser console
3. Review this README
4. Test components individually
