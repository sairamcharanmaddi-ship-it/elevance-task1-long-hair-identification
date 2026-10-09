import os
import joblib
import numpy as np

from deepface import DeepFace
from utils.preprocessing import extract_features

MODEL_PATH = "model/hair_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Hair model not found. Run python train.py first."
    )

hair_model = joblib.load(MODEL_PATH)

def predict_person(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Estimate age and gender presentation
    analysis = DeepFace.analyze(
        img_path=image_path,
        actions=["age", "gender"],
        enforce_detection=True,
        detector_backend="opencv",
    )

    if isinstance(analysis, list):
        analysis = analysis[0]

    age = int(round(float(analysis["age"])))
    baseline_gender = str(analysis["dominant_gender"])

    # Classify hair length with the custom model
    features = extract_features(image_path).reshape(1, -1)

    hair_label = int(hair_model.predict(features)[0])
    hair = "Long Hair" if hair_label == 1 else "Short Hair"

    probabilities = hair_model.predict_proba(features)[0]
    hair_confidence = float(np.max(probabilities)) * 100

    # Apply the internship's task-specific rule
    if 20 <= age <= 30:
        if hair_label == 1:
            output = "Female"
        else:
            output = "Male"

        rule = "Age 20–30: hair-based task rule"

    else:
        # Outside the age range, use the baseline model estimate
        output = baseline_gender
        rule = "Outside age 20–30: baseline estimate"

    return {
        "estimated_age": age,
        "hair": hair,
        "hair_confidence": round(hair_confidence, 2),
        "baseline_gender_estimate": baseline_gender,
        "task_output": output,
        "rule_applied": rule,
    }