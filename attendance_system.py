import cv2
import os
import csv
import pickle
from datetime import datetime
import face_recognition
import numpy as np
from pathlib import Path

class AttendanceSystem:
    def __init__(self):
        self.known_faces_dir = "known_faces"
        self.attendance_file = "attendance.csv"
        self.encodings_file = "face_encodings.pkl"
        
        # Create directories if they don't exist
        Path(self.known_faces_dir).mkdir(exist_ok=True)
        
        # Initialize CSV if it doesn't exist
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Date', 'Time', 'Status'])
        
        # Load known face encodings
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load and encode all known faces from the known_faces directory"""
        print("Loading known faces...")
        
        for person_name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            
            if not os.path.isdir(person_dir):
                continue
            
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                
                # Load image and encode it
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(person_name)
                    print(f"✓ Loaded face of {person_name} from {image_name}")
        
        print(f"Total known faces loaded: {len(self.known_face_encodings)}\n")
    
    def recognize_faces(self, frame):
        """Recognize faces in the given frame"""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find all faces and encodings in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding,
                tolerance=0.6
            )
            name = "Unknown"
            confidence = 0
            
            # Calculate face distances
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, 
                face_encoding
            )
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
            
            face_names.append((name, confidence))
        
        return face_locations, face_names
    
    def mark_attendance(self, name):
        """Mark attendance in CSV file"""
        if name == "Unknown":
            return False
        
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")
        
        # Check if already marked today
        with open(self.attendance_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2 and row[0] == name and row[1] == date:
                    return False  # Already marked
        
        # Add attendance
        with open(self.attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, date, time, 'Present'])
        
        print(f"✓ Attendance marked for {name} at {time}")
        return True
    
    def run(self):
        """Run the attendance system"""
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            print("Error: Could not open camera.")
            return
        
        marked_faces = set()  # Track who has been marked to avoid duplicates
        
        print("Starting Attendance System...")
        print("Press 'q' to quit\n")
        
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to capture frame.")
                break
            
            # Recognize faces
            face_locations, face_names = self.recognize_faces(frame)
            
            # Draw boxes and labels
            for (top, right, bottom, left), (name, confidence) in zip(face_locations, face_names):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Choose color based on recognition
                if name == "Unknown":
                    color = (0, 0, 255)  # Red for unknown
                else:
                    color = (0, 255, 0)  # Green for known
                
                # Draw rectangle
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Draw label with confidence
                label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(
                    frame, 
                    label, 
                    (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )
                
                # Mark attendance
                if name != "Unknown" and name not in marked_faces:
                    if self.mark_attendance(name):
                        marked_faces.add(name)
            
            # Display frame
            cv2.imshow("Attendance System - Face Recognition", frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        camera.release()
        cv2.destroyAllWindows()
        print("\nAttendance system closed.")

if __name__ == "__main__":
    system = AttendanceSystem()
    system.run()
