from flask import Flask, render_template, request, redirect, session
import sqlite3, os, cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.secret_key = "secretkey"
DB = "attendance.db"

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS students(
        student_id TEXT PRIMARY KEY,
        name TEXT,
        password TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS faculty(
        faculty_id TEXT PRIMARY KEY,
        name TEXT,
        password TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS attendance(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT,
        subject TEXT,
        date TEXT,
        time TEXT,
        status TEXT
    )
    """)

    cur.execute("INSERT OR IGNORE INTO students VALUES('S101','Bindu','1234')")
    cur.execute("INSERT OR IGNORE INTO students VALUES('S102','Bindu','1234')")
    cur.execute("INSERT OR IGNORE INTO faculty VALUES('F101','Dr.Rao','admin123')")

    conn.commit()
    conn.close()

init_db()
def run_face_attendance(subject):

    import cv2, os
    import numpy as np
    from datetime import datetime
    import sqlite3

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []
    label_map = {}
    current_label = 0

    # ---------------- CHECK DATASET ----------------
    if not os.path.exists("dataset"):
        print("Dataset not found")
        return None

    # ---------------- TRAIN MODEL ----------------
    for person in os.listdir("dataset"):
        path = f"dataset/{person}"

        if not os.path.isdir(path):
            continue

        label_map[current_label] = person

        for img in os.listdir(path):
            img_path = f"{path}/{img}"
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img_gray is None:
                continue

            img_gray = cv2.resize(img_gray, (200, 200))
            faces.append(img_gray)
            labels.append(current_label)

        current_label += 1

    if len(faces) == 0:
        print("No training data")
        return None

    recognizer.train(faces, np.array(labels))

    # ---------------- CAMERA FIX ----------------
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Camera not opening")
        return None

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    if detector.empty():
        print("Cascade not loaded")
        return None

    # ---------------- FACE RECOGNITION ----------------
    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected = detector.detectMultiScale(gray, 1.1, 4)

        for (x,y,w,h) in detected:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            id_pred, conf = recognizer.predict(face)

            print("Confidence:", conf)  # debug

            if conf < 60:
                matched_sid = label_map[id_pred]

                now = datetime.now()
                today_date = now.strftime("%Y-%m-%d")
                display_date = now.strftime("%d %B %Y")
                current_time = now.strftime("%I:%M %p")

                conn = sqlite3.connect(DB)
                cur = conn.cursor()

                cur.execute("""
                    INSERT INTO attendance(student_id,subject,date,time,status)
                    VALUES(?,?,?,?,?)
                """, (matched_sid, subject, today_date, current_time, "Present"))

                conn.commit()
                conn.close()

                cam.release()
                cv2.destroyAllWindows()

                return (matched_sid, display_date, current_time)

        cv2.imshow("Face Attendance - Press Q to Exit", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    return None

# ---------------- SELECT LOGIN ----------------
@app.route('/')
def select_login():
    return render_template('select_login.html')

# ---------------- STUDENT LOGIN ----------------
@app.route('/student_login', methods=['GET','POST'])
def student_login():
    if request.method == 'POST':
        sid = request.form['student_id']
        pwd = request.form['password']

        conn = sqlite3.connect(DB)
        cur = conn.cursor()
        cur.execute("SELECT name FROM students WHERE student_id=? AND password=?", (sid,pwd))
        user = cur.fetchone()
        conn.close()

        if user:
            session['student_id'] = sid
            session['name'] = user[0]
            return redirect('/dashboard')
        else:
            return "Invalid Student Login"

    return render_template('student_login.html')

# ---------------- STUDENT DASHBOARD ----------------
@app.route('/dashboard')
def dashboard():
    if 'student_id' not in session:
        return redirect('/')
    return render_template('dashboard.html', name=session['name'])

# ---------------- MARK ATTENDANCE ----------------
# ---------------- MARK ATTENDANCE ----------------
@app.route('/mark_attendance')
def mark_attendance():

    if 'student_id' not in session:
        return redirect('/')

    subject = request.args.get('subject') or "Cloud Security"

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []
    label_map = {}
    current_label = 0

    # CHECK DATASET
    if not os.path.exists("dataset"):
        return "❌ Dataset folder not found"

    # TRAIN MODEL
    for person in os.listdir("dataset"):
        path = f"dataset/{person}"

        if not os.path.isdir(path):
            continue

        label_map[current_label] = person

        for img in os.listdir(path):
            img_path = f"{path}/{img}"
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img_gray is None:
                continue

            img_gray = cv2.resize(img_gray, (200, 200))
            faces.append(img_gray)
            labels.append(current_label)

        current_label += 1

    if len(faces) == 0:
        return "❌ No training data found"

    recognizer.train(faces, np.array(labels))

    # CAMERA
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        return "❌ Camera not opening"

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    if detector.empty():
        return "❌ Haarcascade not loaded"

    # ---------------- FACE DETECTION ----------------
    found = False
    detected = []

    for _ in range(20):   # try 20 frames
        ret, img = cam.read()

        if not ret:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected = detector.detectMultiScale(gray, 1.1, 4)

        if len(detected) > 0:
            found = True
            break

    if not found:
        cam.release()
        return "❌ Face Not Detected"

    # PROCESS FACE
    for (x, y, w, h) in detected:
        face_crop = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        id_pred, conf = recognizer.predict(face_crop)

        if conf < 60:
            matched_sid = label_map.get(id_pred)

            conn = sqlite3.connect(DB)
            cur = conn.cursor()

            cur.execute("SELECT name FROM students WHERE student_id=?", (matched_sid,))
            result = cur.fetchone()

            if result:
                matched_name = result[0]
                now = datetime.now()

                today_date = now.strftime("%Y-%m-%d")
                display_date = now.strftime("%d %B %Y")
                current_time = now.strftime("%I:%M %p")

                cur.execute("""
                    INSERT INTO attendance(student_id,subject,date,time,status)
                    VALUES(?,?,?,?,?)
                """, (matched_sid, subject, today_date, current_time, "Present"))

                conn.commit()

                os.makedirs("static/captured", exist_ok=True)
                filename = f"{matched_sid}_{int(now.timestamp())}.jpg"
                cv2.imwrite(f"static/captured/{filename}", img)

                conn.close()
                cam.release()

                return render_template("result.html",
                                       name=matched_name,
                                       sid=matched_sid,
                                       subject=subject,
                                       date=display_date,
                                       time=current_time,
                                       image_filename=filename)

    cam.release()
    return "❌ Face Not Matched"
          
    # ---------------- STUDENT REPORT ----------------
@app.route('/report')
def report():
    if 'student_id' not in session:
        return redirect('/')

    sid = session['student_id']

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
        SELECT subject, COUNT(*)
        FROM attendance
        WHERE student_id=?
        GROUP BY subject
    """, (sid,))

    rows = cur.fetchall()
    conn.close()

    total_classes = 32
    report_data = []

    for subject, attended in rows:
        percentage = round((attended / total_classes) * 100, 1)
        report_data.append((subject, attended, total_classes, percentage))

    return render_template("report.html", data=report_data)

# ---------------- FACULTY LOGIN ----------------
@app.route('/faculty_login', methods=['GET','POST'])
def faculty_login():
    if request.method == 'POST':
        fid = request.form['faculty_id']
        pwd = request.form['password']

        conn = sqlite3.connect(DB)
        cur = conn.cursor()
        cur.execute("SELECT name FROM faculty WHERE faculty_id=? AND password=?", (fid,pwd))
        user = cur.fetchone()
        conn.close()

        if user:
            session['faculty_id'] = fid
            session['faculty_name'] = user[0]
            return redirect('/faculty_dashboard')
        else:
            return "Invalid Faculty Login"

    return render_template('faculty_login.html')

# ---------------- FACULTY DASHBOARD ----------------
@app.route('/faculty_dashboard')
def faculty_dashboard():
    if 'faculty_id' not in session:
        return redirect('/')

    subject_name = "Cloud Security"

    return render_template(
        'faculty_dashboard.html',
        faculty_name=session['faculty_name'],
        subject_name=subject_name
    )

# ---------------- FACULTY TIMETABLE ----------------
@app.route('/faculty_timetable')
def faculty_timetable():
    if 'faculty_id' not in session:
        return redirect('/')

    timetable = {
        "Monday":    ["Cloud Security - CSE1", "Cloud Security - CSE2", "Free"],
        "Tuesday":   ["Cloud Security - CSE3", "Free", "Cloud Security - CSE1"],
        "Wednesday": ["Cloud Security - CSE2", "Cloud Security - CSE3", "Free"],
        "Thursday":  ["Free", "Cloud Security - CSE1", "Cloud Security - CSE2"],
        "Friday":    ["Cloud Security - CSE3", "Free", "Cloud Security - CSE1"]
    }

    hours = [
        "Hour 1 (9:00 - 10:00)",
        "Hour 2 (10:00 - 11:00)",
        "Hour 3 (11:15 - 12:15)"
    ]

    return render_template(
        'faculty_timetable.html',
        timetable=timetable,
        hours=hours
    )

# ---------------- FACULTY ATTENDANCE ----------------
@app.route('/faculty_attendance')
def faculty_attendance():
    if 'faculty_id' not in session:
        return redirect('/')

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT student_id, subject, date, time
        FROM attendance
        WHERE subject='Cloud Security'
        ORDER BY date DESC
    """)
    data = cur.fetchall()
    conn.close()

    return render_template(
        'faculty_attendance.html',
        data=data,
        subject_name="Cloud Security"
    )

# ---------------- FACULTY SUBJECT CONTENT ----------------
@app.route('/faculty_subject_content')
def faculty_subject_content():
    if 'faculty_id' not in session:
        return redirect('/')

    concepts = [
        "Introduction to Cloud Computing",
        "Cloud Service Models (IaaS, PaaS, SaaS)",
        "Cloud Deployment Models",
        "Identity and Access Management (IAM)",
        "Data Encryption in Cloud",
        "Cloud Risk Assessment",
        "Security Compliance Standards",
        "Cloud Incident Response",
        "Zero Trust Security Model",
        "Cloud Security Best Practices"
    ]

    return render_template('faculty_subject_content.html', concepts=concepts)

# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)