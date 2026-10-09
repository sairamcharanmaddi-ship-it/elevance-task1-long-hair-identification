import os
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from utils.preprocessing import extract_features

DATASET_PATH = "dataset/train"
MODEL_PATH = "model/hair_model.pkl"

CLASSES = {
    "short_hair": 0,
    "long_hair": 1,
}

def load_dataset():
    features = []
    labels = []

    for class_name, label in CLASSES.items():
        folder = os.path.join(DATASET_PATH, class_name)

        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Missing dataset folder: {folder}"
            )

        count = 0

        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)

            try:
                vector = extract_features(path)
            except (ValueError, cv2.error) if False else (ValueError,):
                continue

            features.append(vector)
            labels.append(label)
            count += 1

        print(f"{class_name}: {count} images")

    return np.array(features), np.array(labels)

def main():
    X, y = load_dataset()

    if len(X) < 20:
        raise ValueError(
            "Add more images. At least 20 valid images are required."
        )

    if len(np.unique(y)) != 2:
        raise ValueError(
            "Both long_hair and short_hair images are required."
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    print(
        f"\nValidation accuracy: "
        f"{accuracy_score(y_val, predictions) * 100:.2f}%"
    )

    print(
        classification_report(
            y_val,
            predictions,
            labels=[0, 1],
            target_names=["Short Hair", "Long Hair"],
            zero_division=0,
        )
    )

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Saved trained model: {MODEL_PATH}")

if __name__ == "__main__":
    main()