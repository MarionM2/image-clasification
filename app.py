import streamlit as st
import json
import os
import glob
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

# Function to load image file paths from a directory
def load_data(directory_path):
    image_paths = glob.glob(os.path.join(directory_path, '*.jpg'))  # Adjust the file extension if needed
    return image_paths

# Function to save data to a JSON file
def save_to_json(data, filename):
    # Convert non-serializable objects to strings
    serializable_data = [{'file_path': str(image['file_path']), 'cluster': int(image['cluster'])} for image in data if isinstance(image, dict)]
    
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(serializable_data, json_file, ensure_ascii=False, indent=4)

# Function to cluster images using KMeans
def cluster_images(image_paths):
    # Use the file paths as features directly
    features = np.array([[float(i)] for i in range(len(image_paths))])

    # Dummy clustering using KMeans for illustration purposes
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(features)

    # Assign clusters to images
    clustered_images = [{'file_path': file_path, 'cluster': int(cluster)} for file_path, cluster in zip(image_paths, kmeans.labels_)]
    return clustered_images

# Streamlit app
def main():
    st.title("Image Clustering App")

    # Load image file paths from a directory
    directory_path = 'C:/Users/Hp/Downloads/image clasification/shoes'  # Replace with the path to your image directory
    image_paths = load_data(directory_path)

    # Clustering images
    clustered_images = cluster_images(image_paths)

    # Saving clustered data to a JSON file
    save_to_json(clustered_images, 'clustered_data.json')

    # Displaying the clustered images
    for cluster in range(3):  # Assuming 3 clusters for illustration
        st.subheader(f"Cluster {cluster + 1}")
        cluster_image_paths = [image['file_path'] for image in clustered_images if image['cluster'] == cluster]
        for image_path in cluster_image_paths:
            image = Image.open(image_path)
            st.image(image, caption=f"Cluster: {cluster + 1}", use_column_width=True)

if __name__ == '__main__':
    main()
