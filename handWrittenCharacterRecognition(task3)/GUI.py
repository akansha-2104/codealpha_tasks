import tkinter as tk
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw

# Load the trained model
model = load_model("handwritten_character_model.keras")

# Character mapping (update if needed based on dataset)
characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Initialize Tkinter window
window = tk.Tk()
window.title("Handwritten Character Recognition")
window.geometry("300x400")

# Canvas for drawing
canvas = tk.Canvas(window, width=280, height=280, bg="white")
canvas.pack()

# PIL Image for drawing
image = Image.new("L", (280, 280), 255)
draw = ImageDraw.Draw(image)

# Function to draw on canvas
def draw_digit(event):
    x, y = event.x, event.y
    canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill='black', width=5)
    draw.ellipse([x - 10, y - 10, x + 10, y + 10], fill='black')

canvas.bind("<B1-Motion>", draw_digit)

# Preprocessing function
def preprocess_image():
    img = np.array(image)
    img = cv2.bitwise_not(img)  # Invert colors
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)  # Apply thresholding

    # Find bounding box
    coords = cv2.findNonZero(img)
    if coords is None:
        result_label.config(text="Error: No input detected")
        return None
    x, y, w, h = cv2.boundingRect(coords)

    # Ensure minimum size
    if w < 5 or h < 5:
        result_label.config(text="Error: Draw a clearer character")
        return None

    # Crop with padding
    pad = 10
    x, y = max(x - pad, 0), max(y - pad, 0)
    w, h = min(w + 2 * pad, img.shape[1]), min(h + 2 * pad, img.shape[0])
    img = img[y:y+h, x:x+w]

    # Resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize & reshape
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

# Function to predict character
def predict_character():
    img = preprocess_image()
    if img is None:
        return  # Skip prediction if preprocessing failed

    pred_index = np.argmax(model.predict(img))
    predicted_character = characters[pred_index]
    result_label.config(text=f"Predicted: {predicted_character}")

# Function to clear canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, 280, 280], fill="white")

# Buttons
predict_button = tk.Button(window, text="Predict", command=predict_character)
predict_button.pack()

clear_button = tk.Button(window, text="Clear", command=clear_canvas)
clear_button.pack()

# Label for output
result_label = tk.Label(window, text="Draw a digit or letter and press Predict", font=("Arial", 12))
result_label.pack()

window.mainloop()