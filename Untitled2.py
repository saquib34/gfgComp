#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


# In[2]:


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


# In[3]:


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


# In[4]:


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
    # Calculate centroid
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        return False

    # Get distances from centroid to contour points
    distances = []
    for point in contour:
        dist = np.sqrt((point[0] - cx)**2 + (point[1] - cy)**2)
        distances.append(dist)

    # Smooth the distances to reduce noise
    distances = np.convolve(distances, np.ones(5)/5, mode='same')

    # Find peaks and valleys
    peaks = []
    valleys = []
    for i in range(2, len(distances) - 2):
        if distances[i-2] < distances[i-1] < distances[i] > distances[i+1] > distances[i+2]:
            peaks.append(i)
        elif distances[i-2] > distances[i-1] > distances[i] < distances[i+1] < distances[i+2]:
            valleys.append(i)

    # A star should have at least 5 peaks and 5 valleys
    if len(peaks) >= 5 and len(valleys) >= 5:
        # Check if peaks and valleys alternate
        sorted_extrema = sorted(peaks + valleys)
        for i in range(1, len(sorted_extrema)):
            if (sorted_extrema[i] in peaks) == (sorted_extrema[i-1] in peaks):
                return False
        
        # Check the ratio of longest to shortest distance
        max_dist = max(distances)
        min_dist = min(distances)
        if max_dist / min_dist > 1.5:  # Adjust this threshold as needed
            return True

    return False

def is_ellipse(contour):
    if len(contour) < 5:
        return False

    (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
    
    # Calculate the area ratio between the contour and the fitted ellipse
    ellipse_area = np.pi * MA * ma / 4
    contour_area = cv2.contourArea(contour)
    area_ratio = min(ellipse_area, contour_area) / max(ellipse_area, contour_area)

    # Calculate eccentricity
    eccentricity = np.sqrt(1 - (min(MA, ma) / max(MA, ma))**2)

    # An ellipse should have a good area ratio fit and a certain level of eccentricity
    return area_ratio > 0.90 and 0.2 < eccentricity < 0.99


# In[5]:


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


# In[6]:


csv_file=input()
paths_XYs = read_csv(csv_file) 

# In[8]:


print(len(paths_XYs))


# In[14]:


all_completed_images = []

for i in range(len(paths_XYs)):
    # Define the size of the image
    width, height = 400, 400
    # Create a blank white image
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    # Example coordinates (replace this with your actual coordinates)
    coordinates = paths_XYs[i]
    # Flatten the list of coordinates if necessary
    flat_coords = []
    for coord in coordinates:
        if isinstance(coord, np.ndarray) and coord.shape == (2,):
            flat_coords.append(coord)
        elif isinstance(coord, np.ndarray) and len(coord.shape) > 1:
            flat_coords.extend(coord)
        else:
            flat_coords.append(coord)
    # Convert the flattened list to a numpy array of type int32
    points = np.array(flat_coords, np.int32)
    # Reshape the points array to match the format required by cv2.polylines
    points = points.reshape((-1, 1, 2))
    # Draw the shape on the image
    cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)
    # Convert BGR image to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a binary threshold to the grayscale image
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Display the result
    plt.imshow(image_rgb)
    plt.show()
    # Example usage
    contours = np.array([contours])
    print(contours.shape)
    # Assuming contours.shape is (1, 1, N, 1, 2)
    contour = contours[0, 0, :, 0, :]
    shape = detect_shape(contour)
    print(f"The detected shape is: {shape}")
    
    # Detect symmetry and plot the symmetry lines
    symmetry_lines, symmetry_types = detect_symmetry(image)
    # Convert BGR image to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Plot the result with symmetry lines and print the types of symmetry
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axes
    plt.title("Symmetry Detection")
    # Add text annotations for symmetry lines
    for line in symmetry_lines:
        x1, y1, slope = line
        x2 = width - 1
        y2 = int(slope * (x2 - x1) + y1)
        plt.plot([x1, x2], [y1, y2], 'g-', lw=2)  # Plot symmetry line
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        plt.text(mid_x, mid_y, "Reflection", color='green', fontsize=8, ha='center', va='center')
    # Add text annotation for rotation symmetry
    if any("Rotation" in sym_type for sym_type in symmetry_types):
        center_x, center_y = np.mean(points, axis=0)[0]
        plt.text(center_x, center_y, "Rotation", color='blue', fontsize=10, ha='center', va='center')
    plt.show()
    print("Detected symmetry types:")
    for symmetry_type in set(symmetry_types):  # Use set to remove duplicates
        print(symmetry_type)
        
    # Apply curve completion
    completed_image = complete_curves(image)
    # Convert BGR image to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    completed_image_rgb = cv2.cvtColor(completed_image, cv2.COLOR_BGR2RGB)
    # Display original and completed images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(image_rgb)
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(completed_image_rgb)
    ax2.set_title('Completed Curves')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()
    print("Curve completion process finished.")
    
    # Append the completed image to the list
    all_completed_images.append(completed_image)

# Merge all completed images
if all_completed_images:
    # Find the maximum dimensions
    max_height = max(img.shape[0] for img in all_completed_images)
    max_width = max(img.shape[1] for img in all_completed_images)

    # Create a blank canvas
    merged_image = np.zeros((max_height, max_width, 3), dtype=np.uint8)

    # Overlay all images
    for img in all_completed_images:
        # Create a mask for non-black pixels
        mask = np.any(img != [0, 0, 0], axis=-1)
        # Only copy non-black pixels
        merged_image[0:img.shape[0], 0:img.shape[1]][mask] = img[mask]

    # Display the merged image
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB))
    plt.title('Merged Completed Curves')
    plt.axis('off')
    plt.show()
    print("All curves merged and displayed.")
else:
    print("No completed images to merge.")


# In[ ]:




