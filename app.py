import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from osgeo import gdal
from shapely.geometry import Point
import io

# Function to process raster image and detect edges, contours, and corner points
def process_raster_image(tif_path):
    # Open image with GDAL
    ds = gdal.Open(tif_path)
    if ds is None:
        st.error("Failed to open the TIFF file.")
        return None, None

    # Extract georeferencing information
    geotransform = ds.GetGeoTransform()
    if geotransform is None:
        st.error("Failed to get geotransform information.")
        return None, None

    # Extract image data
    image = ds.GetRasterBand(1).ReadAsArray()

    # Normalize image values to [0, 255]
    image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)

    # Detect edges
    edges = cv2.Canny(image, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Approximate polygons
    approx_polygons = []
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx_polygons.append(approx)

    # Identify corner points
    corner_points = []
    for polygon in approx_polygons:
        for point in polygon:
            corner_points.append(tuple(point[0]))  # Extracting the corner points

    # Filter and sort corner points
    corner_points_filtered = []
    for point in corner_points:
        if all(np.linalg.norm(np.array(point) - np.array(existing_point)) > 10 for existing_point in corner_points_filtered):
            corner_points_filtered.append(point)

    corner_points_filtered.sort(key=lambda point: point[1])

    # Convert pixel coordinates to geographic coordinates
    corner_points_geocoords = []
    for point in corner_points_filtered:
        x_geo = geotransform[0] + point[0] * geotransform[1] + point[1] * geotransform[2]
        y_geo = geotransform[3] + point[0] * geotransform[4] + point[1] * geotransform[5]
        corner_points_geocoords.append((x_geo, y_geo))

    return image, corner_points_geocoords, corner_points_filtered

# Function to calculate Euclidean distance
def calculate_euclidean_distance(coords, point1_index, point2_index):
    point1 = coords[point1_index - 1]  # Convert 1-based index to 0-based index
    point2 = coords[point2_index - 1]
    distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    return distance

# Streamlit app UI
def main():
    st.title("Road Network Analysis")

    # Upload an image
    uploaded_file = st.file_uploader("Upload a road network image (PNG, TIFF)", type=["png", "tiff"])
    if uploaded_file is not None:
        # Process the uploaded image
        img_bytes = uploaded_file.read()
        image_path = "uploaded_image.png"
        with open(image_path, "wb") as f:
            f.write(img_bytes)

        st.image(img_bytes, caption="Uploaded Image", use_column_width=True)

        # Process the image
        image, corner_points_geocoords, corner_points_filtered = process_raster_image(image_path)

        if corner_points_geocoords:
            # Display corner points on the image
            st.subheader("Corner Points")
            st.write("The following are the corner points in geographic coordinates:")

            for i, (x_geo, y_geo) in enumerate(corner_points_geocoords, start=1):
                st.write(f"Point {i}: ({x_geo:.2f}, {y_geo:.2f})")

            # Let user select nodes for distance calculation
            st.subheader("Calculate Euclidean Distance between Points")
            point1_index = st.number_input("Select first point:", min_value=1, max_value=len(corner_points_filtered), value=1)
            point2_index = st.number_input("Select second point:", min_value=1, max_value=len(corner_points_filtered), value=2)

            # Calculate and display the Euclidean distance
            distance = calculate_euclidean_distance(corner_points_geocoords, point1_index, point2_index)
            st.write(f"The Euclidean distance between Point {point1_index} and Point {point2_index} is {distance:.2f} units.")

            # Display road network with corner points
            st.subheader("Road Network with Corner Points")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(image, cmap='gray')
            corner_points = np.array(corner_points_filtered)
            ax.scatter(corner_points[:, 0], corner_points[:, 1], c='r', s=50)
            for i, point in enumerate(corner_points_filtered, start=1):
                ax.text(point[0], point[1], str(i), color='b', fontsize=14, ha='center', va='center')
            st.pyplot(fig)
        else:
            st.error("No corner points found.")

if __name__ == "__main__":
    main()
