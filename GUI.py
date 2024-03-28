import cv2 as cv
import numpy as np
import mediapipe as mp
import tkinter as tk
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

mp_face_mesh = mp.solutions.face_mesh

# Irises indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Conversion factor: Replace with the specific conversion factor for your setup
conversion_factor = 0.1  # Example conversion factor (adjust as per your setup)

captured_image_path = "captured_photo.jpg"
diameter_values = []

def take_photo():
    camera = cv.VideoCapture(0)
    _, frame = camera.read()
    camera.release()
    return frame

def process_image():
    global diameter_values
    frame = take_photo()

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [frame_w, frame_h]).astype(int)
                                    for p in results.multi_face_landmarks[0].landmark])

            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            cv.circle(frame, center_left, int(l_radius), (0, 255, 0), 2, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (0, 255, 0), 2, cv.LINE_AA)

            # Calculate pupil diameter in millimeters
            l_diameter_mm = 2 * l_radius * conversion_factor
            r_diameter_mm = 2 * r_radius * conversion_factor

            # Update the diameter values list
            diameter_values.append((l_diameter_mm + r_diameter_mm) / 2)

            # Determine stress condition based on pupil diameter
            if l_diameter_mm >= 2 or r_diameter_mm >= 2:
                condition = "Stress"
            else:
                condition = "Non-Stress"

            # Convert BGR image to RGB for displaying with PIL
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Display the frame with iris detection using Tkinter
            image = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
            label.configure(image=image)
            label.image = image

            info_label.config(text=f"Pupil Detection\nCondition: {condition}\nLeft Diameter: {l_diameter_mm:.2f} mm, Right Diameter: {r_diameter_mm:.2f} mm")

            # Save the image
            cv.imwrite(captured_image_path, frame)
            print(f"Image saved as {captured_image_path}")

def display_image():
    # Display the captured image using OpenCV
    image = cv.imread(captured_image_path)
    cv.imshow("Captured Image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def display_graph():
    # Plot the diameter values
    plt.figure(figsize=(6, 4))
    plt.plot(diameter_values, marker='o', color='b', label='Pupil Diameter')
    plt.xlabel('Frame Number')
    plt.ylabel('Pupil Diameter (mm)')
    plt.title('Pupil Diameter Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Create the main application window
window = tk.Tk()
window.title("Pupil Detection")
window.geometry("800x800")

# Create a label to display the image
label = tk.Label(window)
label.pack()

# Create a button to capture and process the image
capture_button = tk.Button(window, text="Capture Photo", command=process_image)
capture_button.pack()

# Create a button to display the captured image
display_button = tk.Button(window, text="Display Image", command=display_image)
display_button.pack()

# Create a button to display the pupil diameter graph
graph_button = tk.Button(window, text="Show Graph", command=display_graph)
graph_button.pack()

# Create a label to display iris detection information
info_label = tk.Label(window, text="")
info_label.pack()

# Start the Tkinter event loop
window.mainloop()

