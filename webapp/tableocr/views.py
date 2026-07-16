"""Django views for Table OCR webapp."""

import os
import csv
import uuid
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from django.shortcuts import render
from django.http import JsonResponse, FileResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
# declare logger
import logging
logger = logging.getLogger(__name__)

def index(request):
    """Main page."""
    return render(request, 'index.html')


@csrf_exempt
@require_http_methods(["POST"])
def upload_file(request):
    """Handle file upload with optional rotation."""
    logger.info("Uploading file")

    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file provided'}, status=400)

    file = request.FILES['file']
    rotation = int(request.POST.get('rotation', 0))
    logger.info(f"File uploaded with rotation: {rotation}")

    # Validate file extension
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    file_ext = Path(file.name).suffix.lower()

    if file_ext not in allowed_extensions:
        return JsonResponse({
            'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'
        }, status=400)

    # Generate session ID
    session_id = str(uuid.uuid4())

    # Save uploaded file
    upload_dir = Path(settings.MEDIA_ROOT) / 'uploads'
    upload_dir.mkdir(parents=True, exist_ok=True)

    upload_filename = f"{session_id}{file_ext}"
    upload_path = upload_dir / upload_filename

    with open(upload_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    # Apply rotation if needed
    if rotation != 0:
        try:
            import cv2

            # Read image
            img = cv2.imread(str(upload_path))

            if img is None:
                logger.error(f"Failed to read image for rotation: {upload_path}")
            else:
                # Rotate image
                if rotation == 90:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    img = cv2.rotate(img, cv2.ROTATE_180)
                elif rotation == 270:
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Save rotated image
                cv2.imwrite(str(upload_path), img)
                logger.info(f"Image rotated by {rotation} degrees")
        except Exception as e:
            logger.error(f"Error rotating image: {e}")
            # Continue anyway - rotation is optional

    # Clean old files
    cleanup_old_files()

    return JsonResponse({
        'session_id': session_id,
        'message': 'File uploaded successfully',
        'rotation_applied': rotation
    })


def download_csv(request, session_id):
    """Download CSV file."""
    file_path = Path(settings.MEDIA_ROOT) / 'results' / f"{session_id}.csv"

    if not file_path.exists():
        raise Http404("File not found")

    response = FileResponse(open(file_path, 'rb'), content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="extracted_table_{session_id}.csv"'
    return response


def download_image(request, session_id):
    """Download image file."""
    file_path = Path(settings.MEDIA_ROOT) / 'results' / f"{session_id}.png"

    if not file_path.exists():
        raise Http404("File not found")

    response = FileResponse(open(file_path, 'rb'), content_type='image/png')
    response['Content-Disposition'] = f'attachment; filename="visualization_{session_id}.png"'
    return response


def preview_csv(request, session_id):
    """Preview CSV content."""
    file_path = Path(settings.MEDIA_ROOT) / 'results' / f"{session_id}.csv"

    if not file_path.exists():
        return JsonResponse({'error': 'File not found'}, status=404)

    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)

        return JsonResponse({
            'rows': len(data),
            'columns': len(data[0]) if data else 0,
            'data': data
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def preview_image(request, session_id):
    """Serve image for preview."""
    file_path = Path(settings.MEDIA_ROOT) / 'results' / f"{session_id}.png"

    if not file_path.exists():
        raise Http404("File not found")

    return FileResponse(open(file_path, 'rb'), content_type='image/png')


def cleanup_old_files():
    """Remove files older than FILE_RETENTION_HOURS."""
    now = datetime.now()
    cutoff_time = now - timedelta(hours=settings.FILE_RETENTION_HOURS)

    for directory in ['uploads', 'results']:
        dir_path = Path(settings.MEDIA_ROOT) / directory
        if not dir_path.exists():
            continue

        for file_path in dir_path.iterdir():
            if file_path.is_file():
                file_modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_modified_time < cutoff_time:
                    try:
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path.name}")
                    except Exception as e:
                        logger.info(f"Error deleting file {file_path.name}: {e}")
