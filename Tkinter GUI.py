import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from predict import predict_person

class LongHairApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Long Hair Identification")
        self.root.geometry("760x720")
        self.root.minsize(650, 600)

        self.image_path = None
        self.photo = None

        tk.Label(
            root,
            text="LONG HAIR IDENTIFICATION",
            font=("Arial", 21, "bold"),
        ).pack(pady=18)

        tk.Label(
            root,
            text="Upload a person's image to run the task-specific model.",
            font=("Arial", 10),
        ).pack(pady=3)

        self.preview = tk.Label(
            root,
            text="Image Preview",
            width=45,
            height=14,
            relief="groove",
        )
        self.preview.pack(pady=15)

        buttons = tk.Frame(root)
        buttons.pack(pady=8)

        tk.Button(
            buttons,
            text="Upload Image",
            width=15,
            command=self.upload_image,
        ).grid(row=0, column=0, padx=6)

        tk.Button(
            buttons,
            text="Predict",
            width=15,
            command=self.predict,
        ).grid(row=0, column=1, padx=6)

        tk.Button(
            buttons,
            text="Clear",
            width=15,
            command=self.clear,
        ).grid(row=0, column=2, padx=6)

        self.result = tk.Label(
            root,
            text="Prediction results will appear here.",
            font=("Arial", 12),
            justify="left",
            wraplength=650,
        )
        self.result.pack(pady=20, padx=20)

        tk.Label(
            root,
            text=(
                "Demo only: age, gender-presentation, and hair estimates "
                "may be inaccurate."
            ),
            font=("Arial", 9),
            wraplength=680,
        ).pack(side="bottom", pady=12)

    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ],
        )

        if not path:
            return

        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((430, 320))
                self.photo = ImageTk.PhotoImage(image)

            self.image_path = path
            self.preview.configure(image=self.photo, text="")
            self.result.configure(
                text="Image loaded. Click Predict to continue."
            )

        except Exception as exc:
            messagebox.showerror("Image Error", str(exc))

    def predict(self):
        if not self.image_path:
            messagebox.showwarning(
                "No image",
                "Please upload an image first.",
            )
            return

        self.result.configure(text="Processing image...")
        self.root.update_idletasks()

        try:
            result = predict_person(self.image_path)

            display_text = (
                f"Estimated Age: {result['estimated_age']}\n"
                f"Hair Classification: {result['hair']}\n"
                f"Hair Model Score: "
                f"{result['hair_confidence']:.2f}%\n"
                f"Baseline Gender Estimate: "
                f"{result['baseline_gender_estimate']}\n"
                f"Task Output: {result['task_output']}\n"
                f"Rule Applied: {result['rule_applied']}"
            )

            self.result.configure(text=display_text)

        except Exception as exc:
            self.result.configure(text="Prediction failed.")
            messagebox.showerror(
                "Prediction Error",
                f"{exc}\n\nCheck the image and installed models.",
            )

    def clear(self):
        self.image_path = None
        self.photo = None

        self.preview.configure(image="", text="Image Preview")
        self.result.configure(
            text="Prediction results will appear here."
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = LongHairApp(root)
    root.mainloop()