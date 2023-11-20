import tkinter as tk
from PIL import Image, ImageEnhance, ImageTk

combos = []

def create_combo(x, y):
    circle_radius = 20
    square_size = 26

    circle = canvas.create_oval(x - circle_radius, y - circle_radius, x + circle_radius, y + circle_radius, fill="white")
    square_x = x - square_size // 2
    square_y = y - square_size // 2
    square = canvas.create_rectangle(square_x, square_y, square_x + square_size, square_y + square_size, fill="black")

    combo = {
        "circle_x": x,
        "circle_y": y,
        "square_x": square_x,
        "square_y": square_y,
        "circle": circle,
        "square": square,
        "show_coords": False,
    }
    combos.append(combo)

def on_canvas_click(event):
    # Calculate the actual x, y coordinates on the canvas
    x = int(canvas.canvasx(event.x))  # Adjusted x coordinate
    y = int(canvas.canvasy(event.y))  # Adjusted y coordinate

    # Now use the adjusted x, y coordinates for further processing
    create_combo(x, y)

def toggle_coordinates():
    for combo in combos:
        combo["show_coords"] = not combo["show_coords"]
        circle_x = combo["circle_x"]
        circle_y = combo["circle_y"]
        square_x = combo["square_x"]
        square_y = combo["square_y"]

        circle_coords_text = f"({circle_x}, {circle_y})"
        square_coords_text = f"({square_x}, {square_y}"

        if combo["show_coords"]:
            canvas.create_text(circle_x, circle_y - 30, text=circle_coords_text, fill="black")
            canvas.create_text(square_x + 10, square_y + 40, text=square_coords_text, fill="black")
        else:
            canvas.delete("coords_text")

root = tk.Tk()
root.title("Create, Move, Duplicate, and Toggle Coordinates")

# Create a frame for buttons and pack it at the top
button_frame = tk.Frame(root)
button_frame.pack(side=tk.TOP, fill=tk.X)

# Create and pack the Toggle Coordinates button in the button frame
toggle_button = tk.Button(button_frame, text="Toggle Coordinates", command=toggle_coordinates)
toggle_button.pack(side=tk.LEFT)

# Create a frame to contain the canvas and scrollbars
canvas_frame = tk.Frame(root)
canvas_frame.pack(fill=tk.BOTH, expand=True)

# Create canvas with specified size
canvas = tk.Canvas(canvas_frame, bg="beige")
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create vertical scrollbar linked to the canvas
v_scroll = tk.Scrollbar(canvas_frame, orient='vertical', command=canvas.yview)
v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
canvas.configure(yscrollcommand=v_scroll.set)

# Load and process your image with reduced opacity
image = Image.open('Map3_4-5p-resize.jpg')
enhancer = ImageEnhance.Brightness(image)
translucent_image = enhancer.enhance(0.4)
translucent_image = ImageTk.PhotoImage(translucent_image)

# Display the translucent image on the canvas
canvas.create_image(0, 0, anchor=tk.NW, image=translucent_image)

canvas.bind("<Button-1>", on_canvas_click)

# After all elements are created and packed, update the scrollregion
def update_scrollregion():
    canvas.config(scrollregion=canvas.bbox(tk.ALL))

# Bind update_scrollregion to canvas changes
canvas.bind('<Configure>', lambda event: update_scrollregion())

root.mainloop()
