# Image Rotation Feature

## Overview

Added image preview and rotation functionality before OCR extraction. Users can now:
1. Upload/drag-drop an image
2. Preview the image
3. Rotate it 90° clockwise or counterclockwise
4. Confirm rotation before starting extraction

## Changes Made

### 1. Frontend (HTML)

**File:** `tableocr/templates/index.html`

Added new section between upload and processing:

```html
<!-- Image Preview & Rotation Section -->
<section id="preview-section" class="card" style="display: none;">
    <h2>Please ensure the rotation of the table is correct</h2>

    <div class="preview-image-container">
        <img id="preview-image" src="" alt="Uploaded Image">
    </div>

    <div class="rotation-controls">
        <button id="rotate-left-btn">Rotate Left</button>
        <button id="rotate-right-btn">Rotate Right</button>
    </div>

    <div class="rotation-info">
        <p>Current rotation: <span id="rotation-angle">0°</span></p>
    </div>

    <button id="start-extraction-btn">Start Extraction</button>
</section>
```

### 2. Frontend (CSS)

**File:** `tableocr/static/css/styles.css`

Added styles for:
- `.preview-title` - Title with mint color
- `.preview-image-container` - Dark container for image preview
- `.rotation-controls` - Button layout
- `.rotation-btn` - Rotation button styling
- `.rotation-info` - Current rotation display
- `.btn-large` - Large extraction button

### 3. Frontend (JavaScript)

**File:** `tableocr/static/js/app.js`

**New Properties:**
```javascript
this.currentRotation = 0;
this.uploadedFile = null;
```

**New Methods:**

1. **`showImagePreview(file)`** - Displays uploaded image
   - Uses FileReader to read image as data URL
   - Sets preview image source
   - Shows preview section

2. **`rotateImage(degrees)`** - Rotates image by ±90°
   - Updates `currentRotation` (0, 90, 180, 270)
   - Applies CSS transform
   - Updates rotation display

3. **`updateRotationDisplay()`** - Updates UI
   - Applies CSS rotation transform
   - Updates rotation angle text

4. **`startExtraction()`** - Begins processing
   - Called when user clicks "Start Extraction"
   - Uploads file with rotation info

**Modified Methods:**

- **`handleFileSelect(file)`** - Now shows preview instead of uploading directly
- **`uploadFile(file)`** - Sends rotation angle with FormData
- **`showSection(section)`** - Added 'preview' case
- **`resetApp()`** - Resets rotation state

### 4. Backend (Django Views)

**File:** `tableocr/views.py`

**Modified `upload_file(request)`:**

```python
# Get rotation from POST data
rotation = int(request.POST.get('rotation', 0))

# Apply rotation using OpenCV
if rotation != 0:
    import cv2
    img = cv2.imread(str(upload_path))

    if rotation == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rotation == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imwrite(str(upload_path), img)
```

**Returns:** `rotation_applied` in JSON response

## User Flow

```
1. User uploads image (drag-drop or browse)
   ↓
2. Preview section shown with image
   ↓
3. User can rotate left/right (90° increments)
   - Visual rotation updates in real-time
   - Rotation angle displayed (0°, 90°, 180°, 270°)
   ↓
4. User clicks "Start Extraction"
   ↓
5. Image uploaded with rotation parameter
   ↓
6. Server rotates actual image file using OpenCV
   ↓
7. Processing begins with rotated image
```

## Technical Details

### Rotation Handling

**Frontend:**
- Visual rotation via CSS `transform: rotate()`
- Rotation tracked in degrees (0, 90, 180, 270)
- Only visual - doesn't modify file

**Backend:**
- Actual image rotation using OpenCV's `cv2.rotate()`
- Modifies uploaded file before processing
- Supports 90°, 180°, 270° rotations

### OpenCV Rotation Functions

```python
cv2.ROTATE_90_CLOCKWISE      # 90° right
cv2.ROTATE_180               # 180°
cv2.ROTATE_90_COUNTERCLOCKWISE  # 90° left (270°)
```

## UI/UX Features

✅ **Clear instructions** - "Please ensure the rotation of the table is correct"
✅ **Visual feedback** - Live rotation preview
✅ **Rotation indicator** - Shows current angle
✅ **Icon buttons** - Bootstrap icons for rotate arrows
✅ **Large action button** - Prominent "Start Extraction" button
✅ **Consistent theming** - Matches dark charcoal design

## Error Handling

- If OpenCV rotation fails, logs error but continues
- Image validation happens before preview
- Rotation is optional (0° = no rotation)

## Dependencies

**Already available:**
- OpenCV (`cv2`) - Used in main pipeline, already installed

No new dependencies required!

## Testing

1. **Upload image** - Should show preview section
2. **Rotate left** - Image rotates counterclockwise, shows 270°
3. **Rotate right** - Image rotates clockwise, shows 90°
4. **Multiple rotations** - Cycles through 0°, 90°, 180°, 270°
5. **Start extraction** - Processing begins with rotated image
6. **Reset** - Rotation resets to 0° for new upload

## Future Enhancements (Optional)

- [ ] Free rotation (any angle) with slider
- [ ] Flip horizontal/vertical
- [ ] Zoom/pan preview
- [ ] Show original image side-by-side
- [ ] Rotation preview for results too

## Files Modified

1. `tableocr/templates/index.html` - Added preview section
2. `tableocr/static/css/styles.css` - Added preview styles
3. `tableocr/static/js/app.js` - Added rotation logic
4. `tableocr/views.py` - Added server-side rotation

**Total:** 4 files modified, ~200 lines added
