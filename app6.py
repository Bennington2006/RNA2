import streamlit as st
import pandas as pd
import numpy as np
import trimesh
from my_functions6 import (
    extract_rna_spots,
    rotate_rna_spots,
    compute_original_colocalization,
    compute_rotated_colocalization,
    compute_random_colocalization,
    compute_kde_colocalization
)
from multiprocessing import Pool, cpu_count
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit.runtime.scriptrunner")

# Updated rotate_rna_spots function
def rotate_rna_spots(rna_spots, vertices_np):
    """
    Rotate RNA spots based on mesh vertices.
    This function applies a 90-degree clockwise rotation around the z-axis.
    """
    # Calculate the center of the mesh
    mesh_center = vertices_np.mean(axis=0)

    # Translate the RNA spots to the origin
    rna_spots_centered = rna_spots - mesh_center

    # Define the rotation matrix for a 90-degree clockwise rotation around the z-axis
    rotation_matrix_z_clockwise = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])

    # Apply the rotation matrix
    rna_spots_rotated = np.dot(rna_spots_centered, rotation_matrix_z_clockwise.T)

    # Translate the RNA spots back to the original center
    rna_spots_rotated = rna_spots_rotated + mesh_center

    return rna_spots_rotated

# Title and description
st.title("RNA Colocalization Analysis App")
st.write("This app processes RNA colocalization data" \
"")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")
num_trials = st.sidebar.number_input("Number of Trials", min_value=1, value=10, step=1)
coloc_threshold = st.sidebar.number_input("Colocalization Threshold", min_value=0.0, value=0.4, step=0.1)

# File upload section
st.sidebar.header("Upload Files")
uploaded_mesh = st.sidebar.file_uploader("Upload Mesh File (.obj)", type=["obj"])
uploaded_gene1 = st.sidebar.file_uploader("Upload RNA 1 File (.txt)", type=["txt"])
uploaded_gene2 = st.sidebar.file_uploader("Upload RNA 2 File (.txt)", type=["txt"])
uploaded_gene3 = st.sidebar.file_uploader("Upload RNA 3 File (.txt)", type=["txt"])

# Process button
if st.sidebar.button("Run Analysis"):
    if not uploaded_mesh:
        st.error("Please upload the mesh file.")
    else:
        try:
            # Load mesh
            st.write("Loading data...")
            mesh = trimesh.load_mesh(uploaded_mesh, file_type="obj")

            # Load RNA spots (up to 3 files)
            rna_spots_list = []
            for uploaded_file in [uploaded_gene1, uploaded_gene2, uploaded_gene3]:
                if uploaded_file:
                    rna_spots_list.append(extract_rna_spots(uploaded_file))

            if len(rna_spots_list) < 2:
                st.error("Please upload at least 2 RNA files.")
                st.stop()

            # Compute original colocalization
            st.write("Computing original colocalization...")
            avg_original = compute_original_colocalization(rna_spots_list, num_trials, coloc_threshold)
            st.write("### Original Colocalization Results")
            original_table = pd.DataFrame({
                "Type": ["Original"],
                "Average Colocalization (%)": [avg_original]
            })
            st.table(original_table)

            # Compute rotated colocalization
            st.write("Computing rotated colocalization...")
            avg_rotated = compute_rotated_colocalization(rna_spots_list, mesh, num_trials, coloc_threshold)
            st.write("### Rotated Colocalization Results")
            rotated_table = pd.DataFrame({
                "Type": ["Rotated"],
                "Average Colocalization (%)": [avg_rotated]
            })
            st.table(rotated_table)

            # Run random trials
            st.write("Running random trials...")
            avg_random = compute_random_colocalization(rna_spots_list, mesh, num_trials, coloc_threshold)
            st.write("### Randomized Colocalization Results")
            randomized_table = pd.DataFrame({
                "Type": ["Randomized"],
                "Average Colocalization (%)": [avg_random]
            })
            st.table(randomized_table)

            # Run KDE-based trials
            st.write("Running KDE-based trials...")
            avg_kde = compute_kde_colocalization(rna_spots_list, mesh, num_trials, coloc_threshold)
            st.write("### KDE-Based Colocalization Results")
            kde_table = pd.DataFrame({
                "Type": ["KDE-Based"],
                "Average Colocalization (%)": [avg_kde]
            })
            st.table(kde_table)

        except Exception as e:
            st.error(f"An error occurred: {e}")