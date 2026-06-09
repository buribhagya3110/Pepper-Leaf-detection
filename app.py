from base64 import b64decode
from datetime import datetime
from functools import wraps
import mimetypes
import os
import sqlite3
import uuid

from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

from src.predict import predict


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "plant-disease-demo-secret")

UPLOAD_FOLDER = "static/uploads"
DATABASE_PATH = "data/history.sqlite3"
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "password"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

DISEASE_INFO = {
    "Healthy": {
        "disease_name": "Healthy",
        "caused_by": "No disease detected",
        "treatment": "Continue good watering habits, keep leaves dry, and inspect plants regularly.",
        "cure": "No treatment is required. Keep monitoring the plant and maintain preventive care.",
    },
    "Bacterial Spot": {
        "disease_name": "Bacterial Spot",
        "caused_by": "Xanthomonas bacteria spread by splashing water, infected debris, tools, and repeated wet foliage.",
        "treatment": "Remove infected leaves, avoid overhead irrigation, disinfect tools, improve airflow, and use approved copper-based sprays if needed.",
        "cure": "There is no direct cure for infected tissue. Early removal and strict management reduce spread and future damage.",
    },
    "Early Blight": {
        "disease_name": "Early Blight",
        "caused_by": "Alternaria solani fungus — thrives in warm, humid conditions and survives in infected soil and plant debris.",
        "treatment": "Remove and destroy affected leaves, improve air circulation, avoid wetting foliage, apply fungicides (chlorothalonil or copper-based) at first sign.",
        "cure": "No cure for infected tissue. Apply preventive fungicide sprays every 7–10 days during wet seasons and practice crop rotation.",
    },
    "Late Blight": {
        "disease_name": "Late Blight",
        "caused_by": "Phytophthora infestans oomycete — spreads rapidly in cool, moist conditions via air-borne spores.",
        "treatment": "Remove and bag infected plant parts immediately. Apply mancozeb or metalaxyl-based fungicides. Avoid overhead watering.",
        "cure": "No cure once infected. Destroy heavily infected plants. Use resistant varieties and apply preventive copper sprays before wet weather.",
    },
    "Powdery Mildew": {
        "disease_name": "Powdery Mildew",
        "caused_by": "Various fungal species (Leveillula, Erysiphe) — favoured by dry conditions with high humidity and poor airflow.",
        "treatment": "Apply sulfur-based or potassium bicarbonate fungicides. Improve airflow by pruning. Avoid excess nitrogen fertiliser.",
        "cure": "Existing white coating will not disappear, but progression stops with treatment. Remove heavily coated leaves and apply neem oil spray.",
    },
    "Mosaic Virus": {
        "disease_name": "Mosaic Virus",
        "caused_by": "Cucumber Mosaic Virus (CMV) or Tobacco Mosaic Virus (TMV) transmitted by aphids, infected seeds, and contaminated tools.",
        "treatment": "Remove and destroy infected plants immediately. Control aphid vectors with insecticidal soap or reflective mulches. Disinfect all tools.",
        "cure": "There is no cure for viral infections. Prevention through resistant seed varieties, insect control, and strict tool hygiene is essential.",
    },
}


def ensure_storage():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                image_path TEXT NOT NULL,
                prediction TEXT NOT NULL,
                disease_name TEXT NOT NULL,
                caused_by TEXT NOT NULL,
                treatment TEXT NOT NULL,
                cure TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                signals TEXT DEFAULT ''
            )
            """
        )
        # Migrate existing DB: add columns if they don't exist
        try:
            connection.execute("ALTER TABLE history ADD COLUMN confidence REAL DEFAULT 0.0")
        except sqlite3.OperationalError:
            pass  # column already exists
        try:
            connection.execute("ALTER TABLE history ADD COLUMN signals TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass


def db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


def save_uploaded_image(file_storage=None, captured_image=None):
    filename_root = uuid.uuid4().hex

    if file_storage and file_storage.filename:
        original_name = secure_filename(file_storage.filename)
        extension = os.path.splitext(original_name)[1].lower()
        if extension not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
            extension = ".jpg"

        file_name = f"{filename_root}{extension}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file_name)
        file_storage.save(file_path)
        return f"uploads/{file_name}"

    if captured_image:
        header, encoded = captured_image.split(",", 1)
        mime_type = "image/png"
        if ":" in header and ";" in header:
            mime_type = header.split(";")[0].split(":", 1)[1]

        extension = mimetypes.guess_extension(mime_type) or ".png"
        file_name = f"{filename_root}{extension}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file_name)

        with open(file_path, "wb") as image_file:
            image_file.write(b64decode(encoded))

        return f"uploads/{file_name}"

    raise ValueError("No image data was provided.")


def save_history(image_path: str, prediction_result: dict):
    label   = prediction_result["label"]
    conf    = prediction_result.get("confidence", 0.0)
    signals = "; ".join(prediction_result.get("signals", []))

    profile    = DISEASE_INFO.get(label, DISEASE_INFO["Healthy"])
    created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    with db_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO history (
                created_at,
                image_path,
                prediction,
                disease_name,
                caused_by,
                treatment,
                cure,
                confidence,
                signals
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                image_path,
                label,
                profile["disease_name"],
                profile["caused_by"],
                profile["treatment"],
                profile["cure"],
                conf,
                signals,
            ),
        )
        connection.commit()
        return cursor.lastrowid


def get_recent_history(limit=8):
    with db_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, image_path, prediction, disease_name,
                   caused_by, treatment, cure, confidence, signals
            FROM history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("index"))

        error = "Invalid username or password."

    return render_template(
        "login.html",
        error=error,
        default_username=DEFAULT_USERNAME,
        default_password=DEFAULT_PASSWORD,
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    result = None

    if request.method == "POST":
        try:
            file_storage   = request.files.get("file")
            captured_image = request.form.get("captured_image", "").strip()

            if file_storage and file_storage.filename:
                relative_image_path = save_uploaded_image(file_storage=file_storage)
            elif captured_image:
                relative_image_path = save_uploaded_image(captured_image=captured_image)
            else:
                flash("Choose a file or capture a photo from the camera.", "error")
                return redirect(url_for("index"))

            absolute_image_path = os.path.join(app.static_folder, relative_image_path)

            # predict() now returns an enriched dict
            prediction_result = predict(absolute_image_path)
            label   = prediction_result["label"]
            profile = DISEASE_INFO.get(label, DISEASE_INFO["Healthy"])
            save_history(relative_image_path, prediction_result)

            result = {
                "prediction":  label,
                "confidence":  prediction_result["confidence"],
                "signals":     prediction_result["signals"],
                "image_url":   url_for("static", filename=relative_image_path),
                **profile,
            }

            flash("Prediction saved to history.", "success")
        except Exception as error:
            flash(f"Could not process the image: {error}", "error")

    return render_template(
        "index.html",
        username=session.get("username", DEFAULT_USERNAME),
        result=result,
        history=get_recent_history(),
        disease_library=list(DISEASE_INFO.values()),
        default_credentials=f"{DEFAULT_USERNAME} / {DEFAULT_PASSWORD}",
    )


ensure_storage()