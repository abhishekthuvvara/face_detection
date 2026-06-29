# Face Detection & Attendance Register System

An automated attendance system using face recognition that detects and identifies individuals, then logs their attendance in a CSV file.

## Features

- **Face Detection**: Real-time face detection using Haar Cascades
- **Face Recognition**: Identifies known individuals from a database
- **Attendance Logging**: Automatically records attendance with timestamp
- **CSV Export**: Attendance data stored in CSV format
- **Unknown Face Detection**: Highlights unrecognized faces

## System Architecture

```
face_detection/
├── face_detection.py          # Original basic face detection
├── register_face.py           # Register new faces for recognition
├── attendance_system.py       # Main attendance system
├── requirements.txt           # Python dependencies
├── known_faces/              # Directory storing known face images
│   ├── John/
│   │   ├── john_0.jpg
│   │   └── john_1.jpg
│   └── Jane/
│       ├── jane_0.jpg
│       └── jane_1.jpg
└── attendance.csv            # Attendance records
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/abhishekthuvvara/face_detection.git
cd face_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Register Faces

Before the system can recognize people, you need to register their faces:

```bash
python register_face.py
```

**Instructions:**
- Enter the person's name
- Position your face clearly in the frame
- Press `SPACE` to capture each sample (5 samples recommended)
- Press `q` to quit
- The system will store the images in `known_faces/<name>/`

Example:
```
Enter the person's name: John
Number of samples to capture (default 5): 5
✓ Captured sample 1/5
✓ Captured sample 2/5
... (capture more samples)
✓ Successfully registered John with 5 samples!
```

### Step 2: Run Attendance System

```bash
python attendance_system.py
```

**Features:**
- Real-time face recognition
- Green box = Known face
- Red box = Unknown face
- Automatic attendance logging for recognized faces
- Press `q` to stop

### Output: attendance.csv

```
Name,Date,Time,Status
John,2026-06-28,09:15:30,Present
Jane,2026-06-28,09:18:45,Present
John,2026-06-29,09:20:10,Present
```

## How It Works

1. **Registration Phase** (`register_face.py`):
   - Captures multiple face samples for each person
   - Stores images in `known_faces/<name>/` directory
   - Creates a reference database

2. **Recognition Phase** (`attendance_system.py`):
   - Loads all known face encodings
   - Captures video feed from webcam
   - Compares detected faces against known faces
   - Marks attendance for recognized faces
   - Logs to `attendance.csv` with timestamp

3. **Attendance Recording**:
   - One entry per person per day (prevents duplicates)
   - Records Name, Date, Time, and Status
   - Easy to process with Excel or other tools

## Configuration

You can modify these parameters in `attendance_system.py`:

```python
face_detection_tolerance = 0.6  # Lower = stricter matching (default: 0.6)
minNeighbors = 5                 # Haar Cascade sensitivity (default: 5)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not opening | Check camera permissions, ensure no other app uses the camera |
| Poor recognition | Capture more samples (8-10), vary lighting and angles |
| "Unknown" faces detected | Register the person first using `register_face.py` |
| Slow performance | Reduce frame resolution, increase `fx` and `fy` in `recognize_faces()` |

## Hardware Requirements

- Webcam (built-in or USB)
- 4GB RAM minimum
- 500MB disk space

## File Structure

- **face_detection.py**: Original basic face detection (for reference)
- **register_face.py**: Register new faces into the system
- **attendance_system.py**: Main attendance system
- **known_faces/**: Directory containing registered face samples
- **attendance.csv**: Generated attendance log file

## Future Enhancements

- [ ] Web interface for viewing attendance records
- [ ] Database integration (SQLite/PostgreSQL)
- [ ] Email notifications
- [ ] Multiple camera support
- [ ] Liveness detection (prevent spoofing)
- [ ] Deep learning models (ResNet, VGGFace)

## License

This project is open source and available for educational purposes.

## Author

Abhishek Thuvvara

---

**Need help?** Check the code comments or create an issue on GitHub.
