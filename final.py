from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
from typing import Optional, List
import os
import shutil
import re
import math

# -----------------------------
# CONFIG
# -----------------------------
DATABASE_URL = "sqlite:///./medtwin.db"
SECRET_KEY = "change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(title="MedTwin AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# DATABASE MODELS
# -----------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    conditions = Column(Text, default="")

    reports = relationship("MedicalReport", back_populates="user", cascade="all, delete")
    metrics = relationship("HealthMetric", back_populates="user", cascade="all, delete")
    medications = relationship("Medication", back_populates="user", cascade="all, delete")


class MedicalReport(Base):
    __tablename__ = "medical_reports"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    category = Column(String, default="general")
    condition_tag = Column(String, default="")
    file_path = Column(String, nullable=False)
    extracted_text = Column(Text, default="")
    ai_summary = Column(Text, default="")
    serious_flags = Column(Text, default="")
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="reports")


class HealthMetric(Base):
    __tablename__ = "health_metrics"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    metric_type = Column(String, nullable=False)  # bp_systolic, bp_diastolic, glucose
    value = Column(Float, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow)
    note = Column(String, default="")

    user = relationship("User", back_populates="metrics")


class Medication(Base):
    __tablename__ = "medications"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    dosage = Column(String, default="")
    frequency = Column(String, default="")
    adherence_percent = Column(Float, default=100.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="medications")


Base.metadata.create_all(bind=engine)

# -----------------------------
# Pydantic Schemas
# -----------------------------
class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    age: Optional[int] = None
    gender: Optional[str] = None
    conditions: Optional[str] = ""


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MetricCreate(BaseModel):
    metric_type: str
    value: float
    recorded_at: Optional[datetime] = None
    note: Optional[str] = ""


class MedicationCreate(BaseModel):
    name: str
    dosage: Optional[str] = ""
    frequency: Optional[str] = ""
    adherence_percent: Optional[float] = 100.0


class SearchRequest(BaseModel):
    query: str


class SimulationRequest(BaseModel):
    current_glucose: Optional[float] = None
    current_systolic_bp: Optional[float] = None
    exercise_days_per_week_delta: Optional[int] = 0
    sugar_intake_reduction_percent: Optional[float] = 0
    medication_adherence_delta: Optional[float] = 0


# -----------------------------
# HELPERS
# -----------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Form(None), db: Session = Depends(get_db)):
    # This helper is intentionally not used directly; see auth dependency below.
    raise NotImplementedError


def auth_dependency(authorization: Optional[str] = None, db: Session = Depends(get_db)) -> User:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def simple_extract_text(file_path: str) -> str:
    # For a demo, we read plain text-like files. PDFs/images need OCR or parsers later.
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "Report uploaded successfully. Text extraction for this format requires parser integration."


def find_flags(text: str) -> List[str]:
    text_lower = text.lower()
    flags = []
    keywords = {
        "diabetes": ["glucose", "hba1c", "diabetes", "hyperglycemia"],
        "hypertension": ["hypertension", "blood pressure", "systolic", "diastolic"],
        "kidney concern": ["creatinine", "egfr", "kidney"],
        "anemia": ["hemoglobin", "anaemia", "anemia"],
        "liver concern": ["alt", "ast", "bilirubin", "liver"],
    }
    for label, words in keywords.items():
        if any(word in text_lower for word in words):
            flags.append(label)
    return flags


def summarize_report(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "No readable text found."
    short = cleaned[:600]
    flags = find_flags(cleaned)
    flag_text = f" Possible concerns detected: {', '.join(flags)}." if flags else ""
    return f"This report was uploaded and processed. Main extracted content: {short}.{flag_text}"


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def keyword_score(query: str, text: str) -> int:
    q_words = tokenize(query)
    t_words = tokenize(text)
    score = 0
    for word in q_words:
        score += t_words.count(word)
    return score


def health_score(metrics: List[HealthMetric]) -> int:
    if not metrics:
        return 75
    latest = {}
    for m in sorted(metrics, key=lambda x: x.recorded_at):
        latest[m.metric_type] = m.value
    score = 100
    glucose = latest.get("glucose")
    systolic = latest.get("bp_systolic")
    diastolic = latest.get("bp_diastolic")
    if glucose is not None:
        score -= min(abs(glucose - 100) * 0.15, 20)
    if systolic is not None:
        score -= min(abs(systolic - 120) * 0.2, 15)
    if diastolic is not None:
        score -= min(abs(diastolic - 80) * 0.2, 10)
    return max(0, min(100, round(score)))


def metric_trend(values: List[float]) -> str:
    if len(values) < 2:
        return "not enough data"
    if values[-1] > values[0] + 5:
        return "increasing"
    if values[-1] < values[0] - 5:
        return "decreasing"
    return "stable"


# -----------------------------
# AUTH ROUTES
# -----------------------------
@app.post("/register", response_model=TokenResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        name=payload.name,
        email=payload.email,
        password_hash=hash_password(payload.password),
        age=payload.age,
        gender=payload.gender,
        conditions=payload.conditions or "",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_token({"sub": str(user.id)})
    return TokenResponse(access_token=token)


@app.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_token({"sub": str(user.id)})
    return TokenResponse(access_token=token)


# -----------------------------
# PROFILE + DASHBOARD
# -----------------------------
@app.get("/me")
def me(authorization: Optional[str] = None, db: Session = Depends(get_db)):
    user = auth_dependency(authorization, db)
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "age": user.age,
        "gender": user.gender,
        "conditions": user.conditions,
    }


@app.get("/dashboard")
def dashboard(authorization: Optional[str] = None, db: Session = Depends(get_db)):
    user = auth_dependency(authorization, db)
    reports = db.query(MedicalReport).filter(MedicalReport.user_id == user.id).all()
    metrics = db.query(HealthMetric).filter(HealthMetric.user_id == user.id).order_by(HealthMetric.recorded_at.asc()).all()
    meds = db.query(Medication).filter(Medication.user_id == user.id).all()

    glucose_values = [m.value for m in metrics if m.metric_type == "glucose"]
    sys_values = [m.value for m in metrics if m.metric_type == "bp_systolic"]
    dia_values = [m.value for m in metrics if m.metric_type == "bp_diastolic"]

    alerts = []
    if glucose_values and glucose_values[-1] > 140:
        alerts.append("Elevated blood sugar trend detected")
    if sys_values and sys_values[-1] > 140:
        alerts.append("High systolic blood pressure detected")
    if dia_values and dia_values[-1] > 90:
        alerts.append("High diastolic blood pressure detected")

    recs = []
    if glucose_values and metric_trend(glucose_values) == "increasing":
        recs.append("Reduce sugar-heavy foods and track glucose more consistently.")
    if sys_values and metric_trend(sys_values) == "increasing":
        recs.append("Review salt intake, hydration, sleep, and exercise routine.")
    if not recs:
        recs.append("Continue consistent tracking to improve prediction accuracy.")

    return {
        "greeting": f"Welcome back, {user.name}",
        "health_score": health_score(metrics),
        "alerts": alerts,
        "quick_stats": {
            "reports_uploaded": len(reports),
            "medications": len(meds),
            "glucose_trend": metric_trend(glucose_values),
            "bp_systolic_trend": metric_trend(sys_values),
            "bp_diastolic_trend": metric_trend(dia_values),
        },
        "recommendations": recs,
    }


# -----------------------------
# REPORTS + SEARCH
# -----------------------------
@app.post("/reports/upload")
def upload_report(
    title: str = Form(...),
    category: str = Form("general"),
    condition_tag: str = Form(""),
    file: UploadFile = File(...),
    authorization: Optional[str] = None,
    db: Session = Depends(get_db),
):
    user = auth_dependency(authorization, db)

    filename = f"user_{user.id}_{int(datetime.utcnow().timestamp())}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    extracted_text = simple_extract_text(file_path)
    ai_summary = summarize_report(extracted_text)
    serious_flags = ", ".join(find_flags(extracted_text))

    report = MedicalReport(
        user_id=user.id,
        title=title,
        category=category,
        condition_tag=condition_tag,
        file_path=file_path,
        extracted_text=extracted_text,
        ai_summary=ai_summary,
        serious_flags=serious_flags,
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    return {
        "message": "Report uploaded successfully",
        "report_id": report.id,
        "summary": ai_summary,
        "serious_flags": serious_flags.split(", ") if serious_flags else [],
    }


@app.get("/reports")
def list_reports(authorization: Optional[str] = None, db: Session = Depends(get_db)):
    user = auth_dependency(authorization, db)
    reports = db.query(MedicalReport).filter(MedicalReport.user_id == user.id).order_by(MedicalReport.uploaded_at.desc()).all()
    return [
        {
            "id": r.id,
            "title": r.title,
            "category": r.category,
            "condition_tag": r.condition_tag,
            "summary": r.ai_summary,
            "serious_flags": r.serious_flags,
            "uploaded_at": r.uploaded_at,
        }
        for r in reports
    ]


@app.post("/reports/search")
def search_reports(payload: SearchRequest, authorization: Optional[str] = None, db: Session = Depends(get_db)):
    user = auth_dependency(authorization, db)
    reports = db.query(MedicalReport).filter(MedicalReport.user_id == user.id).all()
    scored = []
    for r in reports:
        combined = f"{r.title} {r.category} {r.condition_tag} {r.extracted_text} {r.ai_summary}"
        score = keyword_score(payload.query, combined)
        if score > 0:
            scored.append({
                "id": r.id,
                "title": r.title,
                "category": r.category,
                "condition_tag": r.condition_tag,
                "summary": r.ai_summary,
                "score": score,
            })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {
        "query": payload.query,
        "results": scored,
    }


# -----------------------------
# METRICS + TRENDS
# -----------------------------
@app.post("/metrics")
def add_metric(payload: MetricCreate, authorization: Optional[str] = None, db: Session = Depends(get_db)):
    user = auth_dependency(authorization, db)
    metric = HealthMetric(
        user_id=user.id,
        metric_type=payload.metric_type,
        value=payload.value,
        recorded_at=payload.recorded_at or datetime.utcnow(),
        note=payload.note or "",
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return {"message": "Metric added", "id": metric.id}


@app.get("/trends/{metric_type}")
def get_trend(metric_type: str, authorization: Optional[str] = None, db: Session = Depends(get_db)):
    user = auth_dependency(authorization, db)
    metrics = (
        db.query(HealthMetric)
        .filter(HealthMetric.user_id == user.id, HealthMetric.metric_type == metric_type)
        .order_by(HealthMetric.recorded_at.asc())
        .all()
    )
    values = [m.value for m in metrics]
    avg = round(sum(values) / len(values), 2) if values else None
    return {
        "metric_type": metric_type,
        "trend": metric_trend(values),
        "average": avg,
        "data": [{"value": m.value, "recorded_at": m.recorded_at, "note": m.note} for m in metrics],
    }


# -----------------------------
# MEDICATIONS
# -----------------------------
@app.post("/medications")
def add_medication(payload: MedicationCreate, authorization: Optional[str] = None, db: Session = Depends(get_db)):
    user = auth_dependency(authorization, db)
    med = Medication(
        user_id=user.id,
        name=payload.name,
        dosage=payload.dosage or "",
        frequency=payload.frequency or "",
        adherence_percent=payload.adherence_percent or 100.0,
    )
    db.add(med)
    db.commit()
    db.refresh(med)
    return {"message": "Medication added", "id": med.id}


@app.get("/medications")
def list_meds(authorization: Optional[str] = None, db: Session = Depends(get_db)):
    user = auth_dependency(authorization, db)
    meds = db.query(Medication).filter(Medication.user_id == user.id).order_by(Medication.created_at.desc()).all()
    return [
        {
            "id": m.id,
            "name": m.name,
            "dosage": m.dosage,
            "frequency": m.frequency,
            "adherence_percent": m.adherence_percent,
        }
        for m in meds
    ]


# -----------------------------
# PREDICTION + WHAT-IF
# -----------------------------
@app.get("/predict-risks")
def predict_risks(authorization: Optional[str] = None, db: Session = Depends(get_db)):
    user = auth_dependency(authorization, db)
    metrics = db.query(HealthMetric).filter(HealthMetric.user_id == user.id).order_by(HealthMetric.recorded_at.asc()).all()
    latest = {}
    for m in metrics:
        latest[m.metric_type] = m.value

    risks = []
    glucose = latest.get("glucose")
    sys = latest.get("bp_systolic")
    dia = latest.get("bp_diastolic")

    if glucose is not None and glucose >= 126:
        risks.append({"risk": "possible diabetes concern", "level": "moderate" if glucose < 180 else "high"})
    if sys is not None and sys >= 140:
        risks.append({"risk": "hypertension risk", "level": "moderate" if sys < 160 else "high"})
    if dia is not None and dia >= 90:
        risks.append({"risk": "elevated diastolic blood pressure", "level": "moderate"})

    return {
        "predicted_risks": risks,
        "note": "This is a screening-style educational estimate, not a diagnosis.",
    }


@app.post("/simulate")
def simulate(payload: SimulationRequest):
    glucose = payload.current_glucose
    systolic = payload.current_systolic_bp

    if glucose is not None:
        glucose_change = payload.exercise_days_per_week_delta * 1.8
        glucose_change += (payload.sugar_intake_reduction_percent or 0) * 0.25
        glucose_change += (payload.medication_adherence_delta or 0) * 0.15
        predicted_glucose = max(60, round(glucose - glucose_change, 2))
    else:
        predicted_glucose = None

    if systolic is not None:
        bp_change = payload.exercise_days_per_week_delta * 1.0
        bp_change += (payload.sugar_intake_reduction_percent or 0) * 0.04
        bp_change += (payload.medication_adherence_delta or 0) * 0.08
        predicted_systolic = max(80, round(systolic - bp_change, 2))
    else:
        predicted_systolic = None

    return {
        "predicted_glucose": predicted_glucose,
        "predicted_systolic_bp": predicted_systolic,
        "assumption": "Toy simulation using simple heuristics. Replace with a trained model for real deployment.",
    }


# -----------------------------
# RECOMMENDATIONS
# -----------------------------
@app.get("/recommendations")
def recommendations(authorization: Optional[str] = None, db: Session = Depends(get_db)):
    user = auth_dependency(authorization, db)
    metrics = db.query(HealthMetric).filter(HealthMetric.user_id == user.id).order_by(HealthMetric.recorded_at.asc()).all()
    meds = db.query(Medication).filter(Medication.user_id == user.id).all()

    latest = {}
    for m in metrics:
        latest[m.metric_type] = m.value

    recs = []
    if latest.get("glucose", 100) > 140:
        recs.append("Review sugar intake, meal timing, and regular glucose monitoring.")
    if latest.get("bp_systolic", 120) > 140:
        recs.append("Monitor sodium intake, hydration, and exercise consistency.")
    if meds and any(m.adherence_percent < 80 for m in meds):
        recs.append("Medication adherence appears low. Consider reminders and simpler schedules.")
    if not recs:
        recs.append("Your current tracked metrics look relatively stable. Keep logging data consistently.")

    return {
        "recommendations": recs,
        "disclaimer": "Recommendations are informational and should be reviewed by a clinician.",
    }


@app.get("/")
def home():
    return {"message": "MedTwin AI backend is running"}


# -----------------------------
# RUN
# -----------------------------
# Run locally with:
# uvicorn medtwin_backend_app:app --reload
