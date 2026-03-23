import cv2
import os

# ==============================
# STEP 1: CREATE DATASET FOLDER
# ==============================
os.makedirs("dataset", exist_ok=True)

# ==============================
# STEP 2: ENTER STUDENT ID
# ==============================
student_id = input("Enter Student ID (S101 / S102): ").strip()

# Create folder for student
path = os.path.join("dataset", student_id)
os.makedirs(path, exist_ok=True)

print(f"📁 Folder ready: {path}")

# ==============================
# STEP 3: START CAMERA
# ==============================
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("❌ ERROR: Camera not accessible")
    exit()

print("✅ Camera started")

# ==============================
# STEP 4: LOAD FACE DETECTOR
# ==============================
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if detector.empty():
    print("❌ ERROR: Haarcascade not loaded")
    exit()

# ==============================
# STEP 5: CAPTURE IMAGES
# ==============================
count = 0
print("📸 Capturing faces... Look at camera")

while True:
    ret, frame = cam.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        count += 1
        filename = os.path.join(path, f"{count}.jpg")
        cv2.imwrite(filename, face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Capture Dataset - Press Q to Stop", frame)

    # Stop when Q pressed OR 50 images collected
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

# ==============================
# STEP 6: CLEANUP
# ==============================
cam.release()
cv2.destroyAllWindows()

print(f"✅ SUCCESS: {count} images saved in {path}")