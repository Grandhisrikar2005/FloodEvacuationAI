
from pathlib import Path
from datetime import datetime
import shutil
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
from PIL import Image, ImageTk, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"
IMAGE_DIR = DATASET / "barren_v2" / "Barren"
CSV_PATH = DATASET / "barren_label_review.csv"

CLASSES = [
    "Agriculture", "Barren", "Roads",
    "Urban", "Vegetation", "Water"
]


class LandCoverReviewer:
    def __init__(self, root):
        self.root = root
        self.root.title("FloodEvacuationAI | Land-Cover Review")
        self.root.geometry("1100x780")
        self.root.minsize(850, 650)

        if not CSV_PATH.exists():
            messagebox.showerror(
                "Missing CSV",
                f"Review file not found:\n{CSV_PATH}"
            )
            root.destroy()
            return

        self.df = pd.read_csv(CSV_PATH, keep_default_na=False)

        for column, default in [
            ("original_label", "Barren"),
            ("verified_label", ""),
            ("review_status", "pending"),
            ("review_notes", "")
        ]:
            if column not in self.df.columns:
                self.df[column] = default

        if "filename" not in self.df.columns:
            messagebox.showerror(
                "Invalid CSV", "The CSV needs a filename column."
            )
            root.destroy()
            return

        self.df["filename"] = self.df["filename"].astype(str)
        self.index = 0
        self.dirty = False
        self.photo = None

        self._build_ui()
        self._bind_keys()
        self.show_image()

        root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Satellite Land-Cover Verification",
            font=("Segoe UI", 18, "bold")
        ).pack(anchor="w")

        self.progress_text = ttk.Label(outer, text="")
        self.progress_text.pack(anchor="w", pady=(4, 4))

        self.progress = ttk.Progressbar(
            outer, mode="determinate"
        )
        self.progress.pack(fill="x", pady=(0, 12))

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)

        left = ttk.LabelFrame(body, text="Satellite image", padding=8)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))

        self.image_view = ttk.Label(
            left, text="Loading image...", anchor="center"
        )
        self.image_view.pack(fill="both", expand=True)

        self.image_info = ttk.Label(
            left, text="", justify="left", wraplength=650
        )
        self.image_info.pack(anchor="w", pady=(8, 0))

        right = ttk.Frame(body, width=330)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        info = ttk.LabelFrame(right, text="Image details", padding=10)
        info.pack(fill="x", pady=(0, 10))

        self.filename_label = ttk.Label(
            info, text="", font=("Segoe UI", 11, "bold"),
            wraplength=290
        )
        self.filename_label.pack(anchor="w")

        self.metadata_label = ttk.Label(
            info, text="", justify="left", wraplength=290
        )
        self.metadata_label.pack(anchor="w", pady=(6, 0))

        choices = ttk.LabelFrame(
            right, text="Verified class", padding=10
        )
        choices.pack(fill="x", pady=(0, 10))

        self.label_var = tk.StringVar()

        for number, name in enumerate(CLASSES, 1):
            ttk.Radiobutton(
                choices,
                text=f"{number}. {name}",
                value=name,
                variable=self.label_var,
                command=self.choose_label
            ).pack(anchor="w", pady=2)

        actions = ttk.Frame(choices)
        actions.pack(fill="x", pady=(8, 0))

        ttk.Button(
            actions, text="Clear label", command=self.clear_label
        ).pack(side="left")

        ttk.Button(
            actions, text="Mark uncertain", command=self.uncertain
        ).pack(side="right")

        notes = ttk.LabelFrame(right, text="Review notes", padding=8)
        notes.pack(fill="both", expand=True, pady=(0, 10))

        self.notes = tk.Text(notes, height=5, wrap="word")
        self.notes.pack(fill="both", expand=True)

        status_frame = ttk.LabelFrame(
            right, text="Review status", padding=8
        )
        status_frame.pack(fill="x")

        self.status_var = tk.StringVar(value="pending")
        self.status_combo = ttk.Combobox(
            status_frame,
            textvariable=self.status_var,
            values=["pending", "reviewed", "uncertain"],
            state="readonly"
        )
        self.status_combo.pack(fill="x")
        self.status_combo.bind(
            "<<ComboboxSelected>>", self.status_changed
        )

        nav = ttk.Frame(outer)
        nav.pack(fill="x", pady=(12, 0))

        ttk.Button(
            nav, text="◀ Previous", command=self.previous
        ).pack(side="left")

        ttk.Button(
            nav, text="Next ▶", command=self.next
        ).pack(side="left", padx=6)

        ttk.Button(
            nav, text="Next pending", command=self.next_pending
        ).pack(side="left")

        ttk.Button(
            nav, text="Save progress",
            command=self.save
        ).pack(side="right")

        self.message = ttk.Label(
            outer, text="Ready", relief="sunken", padding=5
        )
        self.message.pack(fill="x", pady=(10, 0))

        ttk.Label(
            outer,
            text="Shortcuts: 1–6 choose class | ←/→ navigate | "
                 "U uncertain | N next pending | Ctrl+S save"
        ).pack(anchor="w", pady=(5, 0))

    def _bind_keys(self):
        self.root.bind("<Control-s>", lambda e: self.save())
        self.root.bind("<Left>", lambda e: self.previous())
        self.root.bind("<Right>", lambda e: self.next())
        self.root.bind("<KeyPress-u>", lambda e: self.uncertain())
        self.root.bind("<KeyPress-n>", lambda e: self.next_pending())

        for number, name in enumerate(CLASSES, 1):
            self.root.bind(
                f"<KeyPress-{number}>",
                lambda e, selected=name: self.set_label(selected)
            )

    def capture(self):
        if self.df.empty:
            return

        self.df.at[self.index, "verified_label"] = (
            self.label_var.get()
        )
        self.df.at[self.index, "review_status"] = (
            self.status_var.get()
        )
        self.df.at[self.index, "review_notes"] = (
            self.notes.get("1.0", "end-1c").strip()
        )

    def show_image(self):
        if self.df.empty:
            self.message.config(text="No images found in CSV.")
            return

        row = self.df.iloc[self.index]
        filename = str(row["filename"])
        path = IMAGE_DIR / filename

        self.filename_label.config(text=filename)

        metadata = []
        for key in [
            "original_label", "location",
            "latitude", "longitude", "zoom"
        ]:
            if key in self.df.columns and str(row.get(key, "")):
                metadata.append(f"{key}: {row[key]}")

        self.metadata_label.config(text="\n".join(metadata))

        if path.exists():
            try:
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(
                        source
                    ).convert("RGB")
                    dimensions = image.size
                    image.thumbnail((720, 510), Image.Resampling.LANCZOS)
                    self.photo = ImageTk.PhotoImage(image)

                self.image_view.config(image=self.photo, text="")
                self.image_info.config(
                    text=f"Image dimensions: {dimensions[0]} × "
                         f"{dimensions[1]} px"
                )
            except Exception as exc:
                self.image_view.config(
                    image="", text=f"Could not load image:\n{exc}"
                )
                self.image_info.config(text=str(path))
        else:
            self.photo = None
            self.image_view.config(
                image="", text=f"IMAGE MISSING\n{path}"
            )
            self.image_info.config(text="Check the image filename.")

        label = str(row["verified_label"])
        self.label_var.set(label if label in CLASSES else "")

        status = str(row["review_status"])
        self.status_var.set(
            status if status in ["pending", "reviewed", "uncertain"]
            else "pending"
        )

        self.notes.delete("1.0", "end")
        self.notes.insert("1.0", str(row["review_notes"]))

        self.update_progress()

    def update_progress(self):
        total = len(self.df)
        reviewed = int((self.df["review_status"] == "reviewed").sum())
        uncertain = int((self.df["review_status"] == "uncertain").sum())
        pending = total - reviewed - uncertain

        self.progress["maximum"] = max(total, 1)
        self.progress["value"] = self.index + 1

        self.progress_text.config(
            text=f"Image {self.index + 1}/{total} | "
                 f"Reviewed: {reviewed} | "
                 f"Uncertain: {uncertain} | Pending: {pending}"
        )

    def set_label(self, label):
        self.label_var.set(label)
        self.status_var.set("reviewed")
        self.capture()
        self.dirty = True
        self.update_progress()

    def choose_label(self):
        self.set_label(self.label_var.get())

    def clear_label(self):
        self.label_var.set("")
        self.status_var.set("pending")
        self.capture()
        self.dirty = True
        self.update_progress()

    def uncertain(self):
        self.label_var.set("")
        self.status_var.set("uncertain")
        self.capture()
        self.dirty = True
        self.update_progress()

    def status_changed(self, _event=None):
        self.capture()
        self.dirty = True
        self.update_progress()

    def navigate(self, index):
        self.capture()
        self.dirty = True
        self.index = index % len(self.df)
        self.show_image()

    def previous(self):
        if len(self.df):
            self.navigate(self.index - 1)

    def next(self):
        if len(self.df):
            self.navigate(self.index + 1)

    def next_pending(self):
        if not len(self.df):
            return

        self.capture()
        total = len(self.df)

        for offset in range(1, total + 1):
            candidate = (self.index + offset) % total
            if self.df.iloc[candidate]["review_status"] != "reviewed":
                self.index = candidate
                self.dirty = True
                self.show_image()
                return

        messagebox.showinfo(
            "Review complete",
            "Every image has been marked reviewed."
        )

    def save(self):
        self.capture()

        try:
            if CSV_PATH.exists():
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup = CSV_PATH.with_name(
                    f"barren_label_review_backup_{stamp}.csv"
                )
                shutil.copy2(CSV_PATH, backup)

            self.df.to_csv(CSV_PATH, index=False)
            self.dirty = False
            self.update_progress()
            self.message.config(text=f"Saved: {CSV_PATH}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def close(self):
        self.capture()

        if self.dirty:
            answer = messagebox.askyesnocancel(
                "Unsaved changes",
                "Save review progress before closing?"
            )
            if answer is None:
                return
            if answer:
                self.save()

        self.root.destroy()


if __name__ == "__main__":
    app_root = tk.Tk()

    try:
        ttk.Style(app_root).theme_use("clam")
    except tk.TclError:
        pass

    LandCoverReviewer(app_root)
    app_root.mainloop()
