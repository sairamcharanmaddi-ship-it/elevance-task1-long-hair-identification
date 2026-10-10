# Long Hair Identification — Elevance Skills Internship

## 1. Project Overview

This project implements an experimental machine-learning application that classifies hair length and applies an age-based decision rule through a graphical user interface.

## 2. Objective

* Train a custom machine-learning model to classify long and short hair.
* Estimate age and gender presentation using a pretrained model.
* Apply the specified task rule for people estimated to be between 20 and 30 years old.
* Provide an easy-to-use Tkinter GUI.
* Include prediction tests and documentation.

## 3. Task Logic

| Estimated age   | Rule                                      |
| --------------- | ----------------------------------------- |
| Below 20        | Use baseline gender-presentation estimate |
| 20–30 inclusive | Long hair → Female; short hair → Male     |
| Above 30        | Use baseline gender-presentation estimate |

The output is an experimental, task-specific label and does not establish a person's actual gender.

## 4. Technologies

* Python
* OpenCV
* NumPy
* Scikit-learn
* DeepFace
* Pillow
* Tkinter
* Joblib

## 5. Project Structure

```text
Task-1-Long-Hair-Identification/
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
├── dataset/
├── model/
├── utils/
└── tests/
```

## 6. Installation

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## 7. Dataset Preparation

Create the following directories:

* `dataset/train/long_hair/`
* `dataset/train/short_hair/`
* `dataset/test/long_hair/`
* `dataset/test/short_hair/`

Use legally obtained, correctly labelled images. Keep the classes balanced and avoid duplicate images across training and testing.

## 8. Train the Model

```bash
python train.py
```

The trained hair classifier is saved as:

`model/hair_model.pkl`

## 9. Launch the GUI

```bash
python app.py
```

Upload an image, preview it, and select Predict.

## 10. Run Tests

```bash
python -m unittest discover -s tests -v
```

## 11. Evaluation

Record validation accuracy, precision, recall, and F1-score. Evaluate on a separate held-out dataset before reporting final results. Do not claim an accuracy value until it has been measured.

## 12. Limitations

* Hair length is not a reliable indicator of gender.
* Age estimation from photographs can be inaccurate.
* The hair classifier is a baseline model and may learn background or lighting patterns.
* Model confidence is not a guarantee of correctness.
* This project is intended for an educational demonstration, not real-world identity decisions.

## 13. Dataset Link

Google Drive dataset: **Add your actual shared Google Drive URL here.**

## 14. Author

Name: Add your name
Program: B.Sc. Data Science
Internship: Elevance Skills
