import tkinter as tk
from tkinter import messagebox, font
import cv2
import pyttsx3
import threading
from PIL import Image, ImageTk
import subprocess

def start_python_processing():
    threading.Thread(target=run_python_file, daemon=True).start()

def run_python_file():
    # 使用 subprocess.Popen 来运行另一个 Python 脚本
    subprocess.Popen(["python", "C:/Users/zbq/Desktop/hackton/final_version.py"])

def load_and_resize_image(path, size):
    # Open an image file
    image = Image.open(path)
    # Resize it to the specified size
    resized_image = image.resize(size, Image.Resampling.LANCZOS)
    # Convert it to a PhotoImage for tkinter use
    return ImageTk.PhotoImage(resized_image)


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()
        cap.release()
        cv2.destroyAllWindows()

def speak():
    text = "Your right elbow is not in the proper position!"
    threading.Thread(target=speak_text, args=(text,), daemon=True).start()
    status_label.config(text="Speaking: " + text)

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Initialize the text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty('rate', 150)

# Setup GUI window
window = tk.Tk()
window.title("Yoga Pose Assistant")
window.geometry("1400x620")  # Set the initial size of the window

# Load and resize the background image
background_image = load_and_resize_image('C:/Users/zbq/Desktop/hackton/lol.png', (1300, 600))

# Create a label for the background
background_label = tk.Label(window, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)  # Fill the entire window


# Use custom font
custom_font = font.Font(family="Helvetica", size=12, weight="bold")

# Create frames
top_frame = tk.Frame(window)
top_frame.pack(fill=tk.X)
bottom_frame = tk.Frame(window)
bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)

# Create a button to start video processing
start_video_button = tk.Button(top_frame, text="Start Video", command=start_python_processing, font=custom_font)
start_video_button.pack(pady=10)

# # Create a button to start speaking
# speak_button = tk.Button(top_frame, text="Speak Correction", command=speak, font=custom_font)
# speak_button.pack(pady=10)

# Status label
status_label = tk.Label(bottom_frame, text="", font=custom_font)
status_label.pack(pady=10, fill=tk.X)

# Set a callback for when the window is closed
window.protocol("WM_DELETE_WINDOW", on_closing)

# Start the GUI event loop
window.mainloop()