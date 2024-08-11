from flask import Flask, request, render_template, send_file, jsonify
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import os
import tempfile
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.backends.backend_svg as svg_backend

app = Flask(__name__)
def detect_symmetry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    symmetry_lines = []
    symmetry_types = []
    
    for contour in contours:
        contour = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        num_vertices = len(contour)
        
        # Check reflection symmetry
        for i in range(num_vertices):
            x1, y1 = contour[i][0]
            x2, y2 = contour[(i + num_vertices // 2) % num_vertices][0]
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                line = (x1, y1, slope)
                if check_symmetric(contour, line):
                    symmetry_lines.append(line)
                    symmetry_types.append("Reflection Symmetry")
        
        # Check rotation symmetry
        for angle in [60, 90, 120, 180]:
            if check_rotation_symmetric(contour, angle):
                symmetry_types.append(f"Rotation Symmetry ({angle}°)")
                
    return symmetry_lines, symmetry_types

def check_symmetric(contour, line):
    reflected_contour = []
    for point in contour:
        x, y = point[0]
        line_x, line_y, slope = line
        # Reflection formula
        x_reflected = (x - slope * y + slope * slope * line_x + line_y) / (1 + slope * slope)
        y_reflected = (y + slope * (x - x_reflected)) / (1 + slope * slope)
        reflected_contour.append((x_reflected, y_reflected))
    reflected_contour = np.array(reflected_contour, dtype=np.int32)
    reflected_contour = reflected_contour.reshape((-1, 1, 2))
    return cv2.matchShapes(reflected_contour, contour, cv2.CONTOURS_MATCH_I3, 0) < 0.1

def check_rotation_symmetric(contour, angle):
    M = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
    rotated_contour = cv2.transform(contour, M)
    return cv2.matchShapes(rotated_contour, contour, cv2.CONTOURS_MATCH_I3, 0) < 0.1
def complete_curves(image):
    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Curve Detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank image for completed curves (only green channel)
    completed_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    for contour in contours:
        # Analyze contour
        if len(contour) < 5:
            continue
        
        # Check if contour is open (has gaps)
        if not cv2.isContourConvex(contour):
            # Fit ellipse to estimate full shape
            if len(contour) >= 5:  # Ellipse fitting requires at least 5 points
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(completed_image, ellipse, 255, 2)
    
    # Convert to 3-channel image with only green channel populated
    green_image = np.zeros_like(image)
    green_image[:,:,1] = completed_image
    
    return green_image

def detect_shape(contour):
    shape_name = "Unidentified"
    contour = contour.reshape(-1, 2)  # Reshape to 2D array of points
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    num_vertices = len(approx)

    # Calculate circularity
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * area / (peri ** 2)

    if num_vertices == 2:
        shape_name = "Line"
    elif is_star_shape(contour):
        shape_name = "Star"
    elif circularity > 0.85:
        shape_name = "Circle"
    elif is_ellipse(contour):
        shape_name = "Ellipse"
    elif num_vertices == 3:
        shape_name = "Triangle"
    elif num_vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            shape_name = "Square"
        else:
            shape_name = "Rectangle"
    elif num_vertices == 5:
        shape_name = "Pentagon"
    elif num_vertices == 6:
        shape_name = "Hexagon"
    else:
        shape_name = f"Polygon with {num_vertices} vertices"

    return shape_name

def is_star_shape(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        return False

    distances = []
    for point in contour:
        dist = np.sqrt((point[0] - cx)**2 + (point[1] - cy)**2)
        distances.append(dist)

    distances = np.convolve(distances, np.ones(5)/5, mode='same')

    peaks = []
    valleys = []
    for i in range(2, len(distances) - 2):
        if distances[i-2] < distances[i-1] < distances[i] > distances[i+1] > distances[i+2]:
            peaks.append(i)
        elif distances[i-2] > distances[i-1] > distances[i] < distances[i+1] < distances[i+2]:
            valleys.append(i)

    if len(peaks) >= 5 and len(valleys) >= 5:
        sorted_extrema = sorted(peaks + valleys)
        for i in range(1, len(sorted_extrema)):
            if (sorted_extrema[i] in peaks) == (sorted_extrema[i-1] in peaks):
                return False
        
        max_dist = max(distances)
        min_dist = min(distances)
        if max_dist / min_dist > 1.5: 
            return True

    return False

def is_ellipse(contour):
    if len(contour) < 5:
        return False

    (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
    
    ellipse_area = np.pi * MA * ma / 4
    contour_area = cv2.contourArea(contour)
    area_ratio = min(ellipse_area, contour_area) / max(ellipse_area, contour_area)

    eccentricity = np.sqrt(1 - (min(MA, ma) / max(MA, ma))**2)

    return area_ratio > 0.90 and 0.2 < eccentricity < 0.99



def create_image(paths_XYs, width=400, height=400):
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    for coordinates in paths_XYs:
        flat_coords = []
        for coord in coordinates:
            if isinstance(coord, np.ndarray) and coord.shape == (2,):
                flat_coords.append(coord)
            elif isinstance(coord, np.ndarray) and len(coord.shape) > 1:
                flat_coords.extend(coord)
            else:
                flat_coords.append(coord)
        points = np.array(flat_coords, np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)
    return image
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    # Save the uploaded file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        file.save(tmp_file.name)
        csv_path = tmp_file.name

    paths_XYs = read_csv(csv_path)

    user_choice = request.form.get('choice')
    all_completed_images = []
    all_figures = []
    svg_filename = None

    for path_XYs in paths_XYs:
        image = create_image(path_XYs)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if user_choice == '1':
            shape = detect_shape(contours[0])
            fig = Figure()
            axis = fig.add_subplot(1, 1, 1)
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axis.set_title(f"Detected Shape: {shape}")
            axis.axis('off')
            all_figures.append(fig)

        elif user_choice == '2':
            symmetry_lines, symmetry_types = detect_symmetry(image)
            fig = Figure()
            axis = fig.add_subplot(1, 1, 1)
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axis.set_title("Symmetry Detection")
            axis.axis('off')
            for line in symmetry_lines:
                x1, y1, slope = line
                x2 = image.shape[1] - 1
                y2 = int(slope * (x2 - x1) + y1)
                axis.plot([x1, x2], [y1, y2], 'g-', lw=2)
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                axis.text(mid_x, mid_y, "Reflection", color='green', fontsize=8, ha='center', va='center')
            if any("Rotation" in sym_type for sym_type in symmetry_types):
                center_x, center_y = np.mean(np.array(contours[0])[:, 0], axis=0)
                axis.text(center_x, center_y, "Rotation", color='blue', fontsize=10, ha='center', va='center')
            all_figures.append(fig)

        elif user_choice == '3':
            shape = detect_shape(contours[0])
            completed_image = complete_curves(image)
            all_completed_images.append(completed_image)
            fig = plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(completed_image, cv2.COLOR_BGR2RGB))
            plt.title('Completed Curves')
            plt.axis('off')
            all_figures.append(fig)

    if user_choice == '3' and all_completed_images:
        max_height = max(img.shape[0] for img in all_completed_images)
        max_width = max(img.shape[1] for img in all_completed_images)
        merged_image = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        for img in all_completed_images:
            mask = np.any(img != [0, 0, 0], axis=-1)
            merged_image[0:img.shape[0], 0:img.shape[1]][mask] = img[mask]
        fig = plt.figure(figsize=(12, 12))
        plt.imshow(cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB))
        plt.title('Merged Completed Curves')
        plt.axis('off')
        all_figures.append(fig)

        # Generate SVG for the merged image
        svg_io = io.BytesIO()
        fig = Figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        canvas = svg_backend.FigureCanvasSVG(fig)
        canvas.draw()
        svg_data = io.BytesIO()
        canvas.print_svg(svg_data)
        svg_data.seek(0)
        svg_filename = 'completed_image.svg'
        with open(svg_filename, 'wb') as f:
            f.write(svg_data.getvalue())

    # Save all figures and send to frontend
    saved_files = []
    for fig in all_figures:
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        encoded_img = base64.b64encode(output.getvalue()).decode('utf-8')
        saved_files.append(f"data:image/png;base64,{encoded_img}")

    # Clean up the temporary file
    os.remove(csv_path)

    response_data = {'images': saved_files}
    if svg_filename:
        response_data['svg_url'] = f'/download_svg/{svg_filename}'

    return jsonify(response_data)

@app.route('/download_svg/<filename>')
def download_svg(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)