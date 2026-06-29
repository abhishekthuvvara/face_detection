import cv2
import os
from pathlib import Path

def capture_face(person_name, num_samples=5):
    """
    Capture multiple face samples for a person to train the recognition model
    
    Args:
        person_name: Name of the person whose face to capture
        num_samples: Number of face samples to capture (default: 5)
    """
    known_faces_dir = "known_faces"
    Path(known_faces_dir).mkdir(exist_ok=True)
    
    person_dir = os.path.join(known_faces_dir, person_name)
    Path(person_dir).mkdir(exist_ok=True)
    
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    print(f"\nCapturing {num_samples} samples for {person_name}")
    print("Position your face clearly in the frame.")
    print("Press SPACE to capture a sample, 'q' to quit\n")
    
    captured_count = 0
    
    while captured_count < num_samples:
        success, frame = camera.read()
        if not success:
            print("Failed to capture frame.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Samples: {captured_count}/{num_samples}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        cv2.imshow("Register Face - Press SPACE to Capture", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"Cancelled. Captured {captured_count} samples.")
            break
        elif key == ord(' '):  # SPACE key
            if len(faces) > 0:
                # Save the frame with the largest detected face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                filename = os.path.join(person_dir, f"{person_name}_{captured_count}.jpg")
                cv2.imwrite(filename, frame)
                captured_count += 1
                print(f"✓ Captured sample {captured_count}/{num_samples}")
            else:
                print("⚠ No face detected. Please reposition your face.")
    
    camera.release()
    cv2.destroyAllWindows()
    
    if captured_count == num_samples:
        print(f"\n✓ Successfully registered {person_name} with {captured_count} samples!")
    else:
        print(f"\n✗ Registration incomplete. Only {captured_count} samples captured.")

if __name__ == "__main__":
    person_name = input("Enter the person's name: ").strip()
    if person_name:
        num_samples = input("Number of samples to capture (default 5): ").strip()
        num_samples = int(num_samples) if num_samples.isdigit() else 5
        capture_face(person_name, num_samples)
    else:
        print("Error: Person name cannot be empty.")
