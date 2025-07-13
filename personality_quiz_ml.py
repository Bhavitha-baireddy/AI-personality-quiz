import tkinter as tk
from tkinter import messagebox
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import random
import time

# Load model & encoder
model = joblib.load("quiz_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Questions & options
questions = [
    "What do you prefer?",
    "You enjoy spending time...",
    "Your biggest fear is...",
    "You prefer...",
    "Your ideal weekend?"
]

original_options = [
    ["Home theatre 📺", "Movie theatre 🎬", "Depends on mood 🎭"],
    ["Alone 🌙", "With friends/family 👨‍👩‍👧‍👦", "Both equally 😊"],
    ["Social crowds 😰", "Being alone too long 😟", "Not being understood 😕"],
    ["Lectures 🎓", "Seminars 🧑‍🏫", "Group discussions 💬"],
    ["Relaxing with books or Netflix 📚", "Party or trip with friends 🎉", "Quiet + social balance 🤝"]
]

class MLPersonalityQuizApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 AI Personality Quiz")
        self.root.geometry("800x600")
        self.root.configure(bg="#f3e5f5")
        self.root.resizable(True, True)

        self.current_q = 0
        self.answers = []
        self.option_map = []  # Stores index mapping for shuffled options

        self.title = tk.Label(root, text="AI Personality Quiz", font=("Helvetica", 24, "bold"),
                              bg="#f3e5f5", fg="#4a148c")
        self.title.pack(pady=20)

        self.question_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="ridge")
        self.question_frame.pack(padx=30, pady=10, fill="both", expand=True)

        self.question_label = tk.Label(self.question_frame, text="", font=("Helvetica", 18),
                                       wraplength=700, justify="left", bg="#ffffff", fg="#4a148c")
        self.question_label.pack(pady=20, padx=20)

        self.radio_var = tk.IntVar()
        self.radio_buttons = []

        for i in range(3):
            rb = tk.Radiobutton(self.question_frame, text="", variable=self.radio_var, value=i,
                                font=("Helvetica", 16), bg="#e1bee7", fg="#212121",
                                activebackground="#ce93d8", selectcolor="#d1c4e9",
                                command=self.auto_next, indicatoron=0, width=40, pady=10)
            rb.pack(pady=8)
            self.radio_buttons.append(rb)

        self.reset_btn = tk.Button(root, text="🔄 Reset Quiz", command=self.reset_quiz,
                                   font=("Helvetica", 14, "bold"), bg="#7e57c2", fg="white", padx=15, pady=5)
        self.reset_btn.pack(pady=15)

        self.load_question()

    def load_question(self):
        self.radio_var.set(-1)
        self.question_label.config(text=f"Q{self.current_q + 1}: {questions[self.current_q]}")

        original = original_options[self.current_q]
        mapped = list(enumerate(original))  # [(0, "opt1"), (1, "opt2"), ...]
        random.shuffle(mapped)  # Shuffle options for UI

        self.option_map = [orig_idx for orig_idx, _ in mapped]  # Keep original indexes

        for i, (_, opt_text) in enumerate(mapped):
            self.radio_buttons[i].config(text=opt_text)

    def auto_next(self):
        selected = self.radio_var.get()
        if selected != -1:
            self.root.after(300, self.next_question)

    def next_question(self):
        selected_ui_index = self.radio_var.get()
        if selected_ui_index == -1:
            messagebox.showwarning("No Selection", "Please select an option.")
            return

        actual_answer_index = self.option_map[selected_ui_index]
        self.answers.append(actual_answer_index)
        self.current_q += 1

        if self.current_q < len(questions):
            self.load_question()
        else:
            self.show_result()

    def reset_quiz(self):
        self.current_q = 0
        self.answers = []
        self.load_question()

    def show_result(self):
        proba = model.predict_proba([self.answers])[0]
        prediction = model.predict([self.answers])[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        all_labels = label_encoder.inverse_transform(np.arange(len(proba)))

        self.root.destroy()
        result_win = tk.Tk()
        result_win.title("🧬 Your Personality Result")
        result_win.geometry("850x600")
        result_win.configure(bg="#fce4ec")
        result_win.resizable(True, True)

        result_title = tk.Label(result_win, text=f"🧠 Your Personality: {predicted_label}",
                                font=("Helvetica", 22, "bold"), bg="#fce4ec", fg="#6a1b9a")
        result_title.pack(pady=25)

        descriptions = {
            "Introvert": "🔵 You are reflective, calm, and value solitude.",
            "Extrovert": "🟢 You are outgoing, talkative, and love social settings.",
            "Ambivert": "🟡 You balance both introversion and extroversion.",
            "Omnivert": "🟣 You adapt based on situations—sometimes loud, sometimes silent."
        }

        desc_label = tk.Label(result_win, text=descriptions[predicted_label],
                              font=("Helvetica", 16), wraplength=750, justify="center",
                              bg="#fce4ec", fg="#4a148c")
        desc_label.pack(pady=10)

        # Prediction confidence chart
        # Fixed label order
        fixed_labels = ["Introvert", "Extrovert", "Ambivert", "Omnivert"]

        # Get probabilities in the fixed order
        proba_dict = dict(zip(label_encoder.inverse_transform(np.arange(len(proba))), proba))
        ordered_proba = [proba_dict.get(label, 0) for label in fixed_labels]

        # Create graph
        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.bar(fixed_labels, ordered_proba, color=["#64b5f6", "#81c784", "#ffd54f", "#ce93d8"])
        ax.set_ylim(0, 1.15)
        ax.set_title("Prediction Confidence", pad=20)
        ax.set_ylabel("Probability")

        for bar, prob in zip(bars, ordered_proba):
            ax.text(bar.get_x() + bar.get_width() / 2, prob + 0.03, f"{prob:.2f}",
                    ha='center', fontsize=10)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=result_win)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

        restart_btn = tk.Button(result_win, text="🔁 Retake Quiz", command=lambda: self.restart(result_win),
                                font=("Helvetica", 14, "bold"), bg="#7e57c2", fg="white", padx=20, pady=5)
        restart_btn.pack(pady=20)

        # Fade in animation
        for i in range(0, 100, 5):
            result_win.attributes("-alpha", i / 100)
            result_win.update()
            time.sleep(0.01)

        result_win.mainloop()

    def restart(self, old_win):
        old_win.destroy()
        root = tk.Tk()
        app = MLPersonalityQuizApp(root)
        root.mainloop()

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = MLPersonalityQuizApp(root)
    root.mainloop()
