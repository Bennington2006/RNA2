import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
import trimesh
import igl
from scipy.spatial import cKDTree
from numba import njit
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rtree import index

# Function to extract RNA spots from text file
def extract_rna_spots(file):
    rna_data = pd.read_csv(file, sep='\t')
    print(rna_data.columns)  # Print the column names to inspect them
    if rna_data.shape[1] < 3:
        raise ValueError("Input file must have at least 3 columns for X, Y, Z positions.")
    rna_spots = rna_data.iloc[:, :3].values
    rna_spots = np.round(rna_spots, 4)
    return rna_spots


# Function to filter spots inside the mesh
def filter_spots_inside_mesh(mesh, spots):
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("The mesh object must be a valid trimesh.Trimesh instance.")
    spots = spots[np.all(np.isfinite(spots), axis=1)]
    min_bound, max_bound = mesh.bounds
    within_bounds = np.all((spots >= min_bound) & (spots <= max_bound), axis=1)
    spots_within_bounds = spots[within_bounds]
    inside = mesh.contains(spots_within_bounds)
    return spots_within_bounds[inside]


# Function to rotate RNA spots
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


def colocper(spots1, spots2, threshold=1):
    if not isinstance(spots1, np.ndarray) or not isinstance(spots2, np.ndarray):
        raise ValueError("spots1 and spots2 must be NumPy arrays.")
    if spots1.shape[1] != 3 or spots2.shape[1] != 3:
        raise ValueError("spots1 and spots2 must have 3 columns (x, y, z).")
    
    distances = cdist(spots1, spots2)
    print("Distances:", distances)  # Debugging: Print distances
    close_spots1 = np.any(distances <= threshold, axis=1)
    percentage_spots1 = np.mean(close_spots1) * 100
    close_spots2 = np.any(distances <= threshold, axis=0)
    percentage_spots2 = np.mean(close_spots2) * 100
    return {'coloc_1_with_2': percentage_spots1, 'coloc_2_with_1': percentage_spots2}


def colocper_three(spots1, spots2, spots3=None, threshold=1):
    if not all(isinstance(spots, np.ndarray) for spots in [spots1, spots2] if spots is not None):
        raise ValueError("All inputs must be NumPy arrays.")

    # Calculate colocalization between spots1 and spots2
    distances_12 = cdist(spots1, spots2)
    coloc_12 = np.any(distances_12 <= threshold, axis=1)
    coloc_spots1 = spots1[coloc_12]

    if coloc_spots1.size == 0:
        return {"coloc_all": 0.0}

    # If spots3 is provided, calculate colocalization with spots3
    if spots3 is not None:
        distances_123 = cdist(coloc_spots1, spots3)
        coloc_123 = np.any(distances_123 <= threshold, axis=1)
        percentage_coloc = np.mean(coloc_123) * 100
    else:
        # If spots3 is not provided, return colocalization between spots1 and spots2
        percentage_coloc = np.mean(coloc_12) * 100

    return {"coloc_all": percentage_coloc}


def colocper_three_e(spots1, spots2, spots3, threshold=1):
    d12 = cdist(spots1, spots2)
    d13 = cdist(spots1, spots3)
    coloc_12 = np.any(d12 <= threshold, axis=1)
    coloc_13 = np.any(d13 <= threshold, axis=1)
    coloc_both = coloc_12 & coloc_13
    coloc_2_only = coloc_12 & ~coloc_13
    coloc_3_only = coloc_13 & ~coloc_12
    total = len(spots1)
    return {
        "coloc_both": 100 * np.sum(coloc_both) / total,
        "coloc_2_only": 100 * np.sum(coloc_2_only) / total,
        "coloc_3_only": 100 * np.sum(coloc_3_only) / total
    }


def coloc_2(spots1, spots2, threshold=1):
    """
    Calculate colocalization between two RNA spot sets (spots1 and spots2).
    This function mimics the behavior of coloc_all but only considers the first two RNA spot sets.
    
    Args:
        spots1 (np.ndarray): RNA spot coordinates for the first set.
        spots2 (np.ndarray): RNA spot coordinates for the second set.
        threshold (float): Distance threshold for colocalization.

    Returns:
        dict: A dictionary containing the percentage of colocalization.
    """
    if not isinstance(spots1, np.ndarray) or not isinstance(spots2, np.ndarray):
        raise ValueError("spots1 and spots2 must be NumPy arrays.")
    if spots1.shape[1] != 3 or spots2.shape[1] != 3:
        raise ValueError("spots1 and spots2 must have 3 columns (x, y, z).")

    # Calculate pairwise distances between spots1 and spots2
    distances = cdist(spots1, spots2)

    # Determine colocalization for spots1 with spots2
    coloc_12 = np.any(distances <= threshold, axis=1)

    # Calculate the percentage of colocalized spots
    percentage_coloc = np.mean(coloc_12) * 100

    return {"coloc_2": percentage_coloc}


def generate_random_spots(mesh, desired_points, oversample_factor=2):
    """
    Generate random points inside the mesh.
    Args:
        mesh (trimesh.Trimesh): The mesh object representing the cell.
        desired_points (int): Number of points to generate (equal to the number of original RNA spots).
        oversample_factor (int): Factor to oversample points for better coverage.

    Returns:
        np.ndarray: Random points inside the mesh.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("The mesh object must be a valid trimesh.Trimesh instance.")
    
    # Get the bounding box of the mesh
    min_bound, max_bound = mesh.bounds

    # Initialize a list to store points inside the mesh
    points_inside_mesh = []

    # Use an independent RNG instance
    rng = np.random.default_rng()

    # Generate points until we have the desired number
    while len(points_inside_mesh) < desired_points:
        # Generate more points than needed to ensure sufficient coverage
        num_points_to_generate = (desired_points - len(points_inside_mesh)) * oversample_factor
        random_points = rng.uniform(low=min_bound, high=max_bound, size=(num_points_to_generate, 3))

        # Check which points are inside the mesh
        inside = mesh.contains(random_points)

        # Add the points inside the mesh to the list
        points_inside_mesh.extend(random_points[inside])

    # Return exactly the desired number of points
    return np.array(points_inside_mesh[:desired_points])


def generate_random_spots_with_rtree(mesh, desired_points, oversample_factor=2):
    """
    Generate random points inside the mesh using Rtree for spatial indexing.
    Args:
        mesh (trimesh.Trimesh): The mesh object representing the cell.
        desired_points (int): Number of points to generate (equal to the number of original RNA spots).
        oversample_factor (int): Factor to oversample points for better coverage.

    Returns:
        np.ndarray: Random points inside the mesh.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("The mesh object must be a valid trimesh.Trimesh instance.")
    
    # Get the bounding box of the mesh
    min_bound, max_bound = mesh.bounds

    # Create an Rtree index for the mesh triangles
    idx = index.Index()
    for i, triangle in enumerate(mesh.triangles):
        idx.insert(i, triangle.bounds)

    # Initialize a list to store points inside the mesh
    points_inside_mesh = []

    # Generate points until we have the desired number
    while len(points_inside_mesh) < desired_points:
        # Generate more points than needed to ensure sufficient coverage
        num_points_to_generate = (desired_points - len(points_inside_mesh)) * oversample_factor
        rng = np.random.default_rng()
        random_points = rng.uniform(low=min_bound, high=max_bound, size=(num_points_to_generate, 3))

        # Check which points are inside the mesh using Rtree
        for point in random_points:
            # Find triangles that might contain the point
            possible_triangles = list(idx.intersection((point[0], point[1], point[2], point[0], point[1], point[2])))
            if any(mesh.contains([point])):
                points_inside_mesh.append(point)

    # Return exactly the desired number of points
    return np.array(points_inside_mesh[:desired_points])


def calculate_max_density(rna_spots, mesh, resolution=1.0):
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("The mesh object must be a valid trimesh.Trimesh instance.")
    min_bound, max_bound = mesh.bounds
    num_bins = np.ceil((max_bound - min_bound) / resolution).astype(int)
    density_grid = np.zeros(num_bins, dtype=int)
    shifted_spots = rna_spots - min_bound
    bin_indices = np.floor(shifted_spots / resolution).astype(int)
    valid_indices = (bin_indices[:, 0] >= 0) & (bin_indices[:, 1] >= 0) & (bin_indices[:, 2] >= 0)
    valid_indices &= (bin_indices[:, 0] < num_bins[0]) & (bin_indices[:, 1] < num_bins[1]) & (bin_indices[:, 2] < num_bins[2])
    bin_indices = bin_indices[valid_indices]
    np.add.at(density_grid, tuple(bin_indices.T), 1)
    density_grid = density_grid / (resolution ** 3)
    return np.max(density_grid)


def generate_kde_spots3(rna_spots, mesh):
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("The mesh object must be a valid trimesh.Trimesh instance.")
    
    kde_model = gaussian_kde(rna_spots.T)
    min_bound, max_bound = mesh.bounds
    bbox_volume = np.prod(max_bound - min_bound)
    spots_per_cubi_um = calculate_max_density(rna_spots, mesh)
    desired_points = int(bbox_volume * spots_per_cubi_um)
    
    # Use an independent RNG instance
    rng = np.random.default_rng()
    sample_space = rng.uniform(low=min_bound, high=max_bound, size=(desired_points, 3))
    
    V = np.array(mesh.vertices)
    F = np.array(mesh.faces)
    W = igl.fast_winding_number_for_meshes(V, F, sample_space)
    inside_mask = W >= 0.5
    points_inside_mesh = sample_space[inside_mask]
    
    if points_inside_mesh.size == 0:
        raise ValueError("No points inside the mesh. Check the mesh or sampling process.")
    
    kde_eval = kde_model(points_inside_mesh.T)
    probabilities = kde_eval / np.sum(kde_eval)
    sampled_indices = rng.choice(len(points_inside_mesh), len(rna_spots), replace=True, p=probabilities)
    return points_inside_mesh[sampled_indices]


def generate_kde_spots4(rna_spots, mesh):
    kde_model = gaussian_kde(rna_spots.T)
    min_bound, max_bound = mesh.bounds
    bbox_volume = np.prod(max_bound - min_bound)
    spots_per_cubi_um = calculate_max_density(rna_spots, mesh)
    desired_points = int(bbox_volume * spots_per_cubi_um)
    sample_space = np.random.uniform(low=min_bound, high=max_bound, size=(desired_points, 3))
    V = np.array(mesh.vertices)
    F = np.array(mesh.faces)
    W = igl.fast_winding_number_for_meshes(V, F, sample_space)
    inside_mask = W >= 0.5
    points_inside_mesh = sample_space[inside_mask]
    kde_eval = kde_model(points_inside_mesh.T)
    probabilities = kde_eval / np.sum(kde_eval)
    sampled_indices = np.random.choice(len(points_inside_mesh), len(rna_spots), replace=True, p=probabilities)
    return points_inside_mesh[sampled_indices], points_inside_mesh, probabilities


def run_random_trial(args):
    mesh, rna_spots_1, rna_spots_2, coloc_threshold = args
    try:
        random_rna_spots_1 = generate_random_spots(mesh, desired_points=len(rna_spots_1))
        random_rna_spots_2 = generate_random_spots(mesh, desired_points=len(rna_spots_2))
        return colocper(random_rna_spots_1, random_rna_spots_2, threshold=coloc_threshold)
    except Exception as e:
        print(f"Error in run_random_trial: {e}. Args: {args}")
        return None


def run_random_trial_triple(args):
    """
    Perform a random trial for colocalization with up to three RNA spot sets.
    Args:
        args (tuple): Contains mesh, spots1, spots2, spots3, and threshold.

    Returns:
        dict: Colocalization results and random spots for debugging.
    """
    mesh, spots1, spots2, spots3, threshold = args

    try:
        # Generate random spots for spots1 and spots2
        random_spots1 = generate_random_spots(mesh, len(spots1))
        random_spots2 = generate_random_spots(mesh, len(spots2))

        if spots3 is None:
            # Perform random trial for 2 RNA spot files
            coloc_random = colocper(random_spots1, random_spots2, threshold=threshold)
            return {"coloc_all": coloc_random['coloc_1_with_2'], "random_spots1": random_spots1, "random_spots2": random_spots2}
        else:
            # Generate random spots for spots3
            random_spots3 = generate_random_spots(mesh, len(spots3))
            coloc_random = colocper_three(random_spots1, random_spots2, random_spots3, threshold=threshold)
            return {"coloc_all": coloc_random['coloc_all'], "random_spots1": random_spots1, "random_spots2": random_spots2, "random_spots3": random_spots3}
    except Exception as e:
        print(f"Error in run_random_trial_triple: {e}")
        return {"coloc_all": 0.0}


def run_random_trial_triple_e(args):
    mesh, rna_spots_1, rna_spots_2, rna_spots_3, coloc_threshold = args
    try:
        random_rna_spots_1 = generate_random_spots(mesh, desired_points=len(rna_spots_1))
        random_rna_spots_2 = generate_random_spots(mesh, desired_points=len(rna_spots_2))
        random_rna_spots_3 = generate_random_spots(mesh, desired_points=len(rna_spots_3))
        return colocper_three_e(random_rna_spots_1, random_rna_spots_2, random_rna_spots_3, threshold=coloc_threshold)
    except Exception as e:
        print(f"Error in run_random_trial_triple_e: {e}. Args: {args}")
        return None


def run_kde_trial(args):
    mesh, rna_spots_1, rna_spots_2, coloc_threshold = args
    try:
        kde_rna_spots_1 = generate_kde_spots3(rna_spots_1, mesh)
        kde_rna_spots_2 = generate_kde_spots3(rna_spots_2, mesh)
        return colocper(kde_rna_spots_1, kde_rna_spots_2, threshold=coloc_threshold)
    except Exception as e:
        print(f"Error in run_kde_trial: {e}. Args: {args}")
        return None


def run_kde_trial_triple(args):
    mesh, rna_spots_1, rna_spots_2, rna_spots_3, coloc_threshold = args
    try:
        kde_rna_spots_1 = generate_kde_spots3(rna_spots_1, mesh)
        kde_rna_spots_2 = generate_kde_spots3(rna_spots_2, mesh)

        if rna_spots_3 is None:
            # Perform KDE-based trial for 2 RNA spot files
            coloc_result = colocper(kde_rna_spots_1, kde_rna_spots_2, threshold=coloc_threshold)
            return {"coloc_all": coloc_result['coloc_1_with_2']}
        else:
            # Perform KDE-based trial for 3 RNA spot files
            kde_rna_spots_3 = generate_kde_spots3(rna_spots_3, mesh)
            coloc_result = colocper_three(kde_rna_spots_1, kde_rna_spots_2, kde_rna_spots_3, threshold=coloc_threshold)
            return {"coloc_all": coloc_result['coloc_all']}
    except Exception as e:
        print(f"Error in run_kde_trial_triple: {e}. Args: {args}")
        return {"coloc_all": 0.0}  # Return 0.0 if an error occurs


def run_kde_trial_triple_e(args):
    mesh, rna_spots_1, rna_spots_2, rna_spots_3, coloc_threshold = args
    try:
        kde_rna_spots_1 = generate_kde_spots3(rna_spots_1, mesh)
        kde_rna_spots_2 = generate_kde_spots3(rna_spots_2, mesh)
        kde_rna_spots_3 = generate_kde_spots3(rna_spots_3, mesh)
        return colocper_three_e(kde_rna_spots_1, kde_rna_spots_2, kde_rna_spots_3, threshold=coloc_threshold)
    except Exception as e:
        print(f"Error in run_kde_trial_triple_e: {e}. Args: {args}")
        return None


def compute_original_colocalization(rna_spots_list, num_trials, coloc_threshold):
    """
    Compute the average original colocalization over multiple trials.
    """
    original_results = []
    for _ in range(num_trials):
        if len(rna_spots_list) == 2:
            colocog = coloc_2(rna_spots_list[0], rna_spots_list[1], threshold=coloc_threshold)
        elif len(rna_spots_list) == 3:
            colocog = colocper_three(rna_spots_list[0], rna_spots_list[1], rna_spots_list[2], threshold=coloc_threshold)
        original_results.append(colocog['coloc_2'] if 'coloc_2' in colocog else colocog['coloc_all'])
    return np.mean(original_results)


def compute_rotated_colocalization(rna_spots_list, mesh, num_trials, coloc_threshold):
    """
    Compute the average rotated colocalization over multiple trials.
    """
    rotated_results = []
    for _ in range(num_trials):
        rotated_spots_1 = rotate_rna_spots(rna_spots_list[0], np.asarray(mesh.vertices))
        if len(rna_spots_list) == 2:
            coloc_rotated = coloc_2(rotated_spots_1, rna_spots_list[1], threshold=coloc_threshold)
        elif len(rna_spots_list) == 3:
            coloc_rotated = colocper_three(rotated_spots_1, rna_spots_list[1], rna_spots_list[2], threshold=coloc_threshold)
        rotated_results.append(coloc_rotated['coloc_2'] if 'coloc_2' in coloc_rotated else coloc_rotated['coloc_all'])
    return np.mean(rotated_results)


def compute_random_colocalization(rna_spots_list, mesh, num_trials, coloc_threshold):
    """
    Compute the average randomized colocalization over multiple trials.
    """
    args = []
    if len(rna_spots_list) == 2:
        args = [(mesh, rna_spots_list[0], rna_spots_list[1], None, coloc_threshold) for _ in range(num_trials)]
    elif len(rna_spots_list) == 3:
        args = [(mesh, rna_spots_list[0], rna_spots_list[1], rna_spots_list[2], coloc_threshold) for _ in range(num_trials)]

    with Pool(processes=cpu_count()) as pool:
        random_results = pool.map(run_random_trial_triple, args)

    random_results = [result['coloc_all'] for result in random_results if result is not None]
    if not random_results:
        raise ValueError("All random trials failed.")
    return np.mean(random_results)


def compute_kde_colocalization(rna_spots_list, mesh, num_trials, coloc_threshold):
    """
    Compute the average KDE-based colocalization over multiple trials.
    """
    args = []
    if len(rna_spots_list) == 2:
        args = [(mesh, rna_spots_list[0], rna_spots_list[1], None, coloc_threshold) for _ in range(num_trials)]
    elif len(rna_spots_list) == 3:
        args = [(mesh, rna_spots_list[0], rna_spots_list[1], rna_spots_list[2], coloc_threshold) for _ in range(num_trials)]

    with Pool(processes=cpu_count()) as pool:
        kde_results = pool.map(run_kde_trial_triple, args)

    kde_results = [result['coloc_all'] for result in kde_results if result is not None and 'coloc_all' in result]
    if not kde_results:
        raise ValueError("All KDE trials failed or returned invalid results.")
    return np.mean(kde_results)


def visualize_random_points(mesh, random_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the mesh vertices
    ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], s=1, color='gray', label='Mesh')

    # Plot the random points
    ax.scatter(random_points[:, 0], random_points[:, 1], random_points[:, 2], s=10, color='red', label='Random Points')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # Example usage
    # Define or load a mesh and RNA spots for testing
    # Replace these with actual test data
    mesh = trimesh.load_mesh("path_to_mesh.obj")  # Replace with the actual mesh file path
    original_rna_spots = np.random.rand(100, 3)  # Replace with actual RNA spots

    random_points = generate_random_spots(mesh, desired_points=len(original_rna_spots))
    assert np.all(mesh.contains(random_points)), "Some random points are outside the mesh."
    assert len(random_points) == len(original_rna_spots), "Number of random points does not match the original RNA spots."
    visualize_random_points(mesh, random_points)