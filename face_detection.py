import cv2

# 1. Load the Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 2. Initialize the webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: could not open camera.")
    exit()

# 3. Main loop for real-time detection
while True:
    success, frame = camera.read()
    if not success:
        print("Failed to capture frame.")
        break
        
    # Convert frame to grayscale for the face detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    
    # Draw rectangles and text around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            "Face Detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
    # Display the output window
    cv2.imshow("Face Detection using python", frame)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 4. Clean up and close windows
camera.release()
cv2.destroyAllWindows()