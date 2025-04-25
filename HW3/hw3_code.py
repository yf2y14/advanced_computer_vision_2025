import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import time

# --- Configuration ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
    print("Warning: __file__ not defined, using current working directory as script base.")
    print(f"Script Base Directory (fallback): {SCRIPT_DIR}")

INPUT_FILENAME = "lena.bmp"
OUTPUT_SUBFOLDER = "output_images"
OUTPUT_FOLDER_PATH = os.path.join(SCRIPT_DIR, OUTPUT_SUBFOLDER)
INPUT_IMAGE_PATH = os.path.join(SCRIPT_DIR, INPUT_FILENAME)
COMPOSITE_IMAGE_NAME = "results_table_with_titles.png"

# Modified parameters for more visible arrows
SHIFT_AMOUNT = 1.0        # Increase the shift amount for larger motion
FLOW_VIS_SCALE = 20       # Double the scale to make arrows longer
FLOW_VIS_STEP = 30       # Smaller step size for more arrows
FLOW_LINE_THICKNESS = 1   # Make lines thicker (3px)
FLOW_LINE_COLOR = (0, 0, 255) 

# Ensure order for table: Iterations increase across columns, Lambda increases down rows
ITERATION_SET = [1, 4, 16, 64]  # Columns
LAMBDA_SET = [0.1, 1.0, 10.0]   # Rows
TABLE_ROWS = len(LAMBDA_SET)
TABLE_COLS = len(ITERATION_SET)

# --- Utility Functions ---

def load_image_grayscale(path):
    """Loads an image, converts to grayscale float64 [0, 255]."""
    print(f"Attempting to load image: {path}")
    if not os.path.exists(path):
        print(f"Error: Input image '{path}' not found.")
        return None
    try:
        img = plt.imread(path)
        img_float = img.astype(np.float64)
        if img_float.ndim == 3 and img_float.shape[2] >= 3:
            print("Converting RGB image to grayscale.")
            gray_img = (0.299 * img_float[:,:,0] +
                        0.587 * img_float[:,:,1] +
                        0.114 * img_float[:,:,2])
        elif img_float.ndim == 2:
             gray_img = img_float
        else:
             print(f"Error: Unsupported image format or dimensions: {img.shape}")
             return None
        print(f"Image loaded successfully: shape={gray_img.shape}")
        return gray_img
    except Exception as e:
        print(f"Error loading image '{path}': {e}")
        return None

def load_result_image(full_path):
    """Loads a previously saved result image (likely color)."""
    print(f"Loading result image: {full_path}")
    if not os.path.exists(full_path):
        print(f"Error: Result image '{full_path}' not found for table.")
        return None
    try:
        img = plt.imread(full_path)
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        if img.ndim == 2:
             img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:
             img = img[:, :, :3]
        return img
    except Exception as e:
        print(f"Error loading result image '{full_path}': {e}")
        return None

def save_image(img_data, full_path):
    """Saves image data using matplotlib to the specified full path."""
    try:
        output_dir = os.path.dirname(full_path)
        if not os.path.exists(output_dir):
             try:
                 os.makedirs(output_dir)
                 print(f"Created directory: {output_dir}")
             except OSError as e:
                 print(f"Error creating directory '{output_dir}': {e}")
                 return False

        if img_data.dtype == np.float64 or img_data.dtype == np.float32:
             img_to_save = np.clip(img_data, 0, 255).astype(np.uint8)
        elif img_data.dtype == np.uint8:
             img_to_save = img_data
        else:
             print(f"Warning: Unknown image data type for saving: {img_data.dtype}. Attempting conversion.")
             img_to_save = np.clip(img_data, 0, 255).astype(np.uint8)

        cmap_param = 'gray' if img_to_save.ndim == 2 else None
        # Use plt.imsave for direct saving without axes, etc.
        plt.imsave(full_path, img_to_save, cmap=cmap_param)
        # Note: If saving the whole figure later, fig.savefig is used instead.
        print(f"Saved image to: {full_path}")
        return True
    except Exception as e:
        print(f"Error saving image to '{full_path}': {e}")
        return False

def shift_image_optimized(img, shift_x, shift_y):
    """Shifts an image using NumPy's built-in functions for better performance."""
    print(f"Applying optimized shift: dx={shift_x}, dy={shift_y}")
    height, width = img.shape
    shifted_img = np.zeros_like(img, dtype=img.dtype)
    
    # Convert shifts to integers for array indexing
    shift_x_int = int(shift_x)
    shift_y_int = int(shift_y)
    
    # Calculate source and destination regions
    src_y_start = max(0, -shift_y_int)
    src_y_end = min(height, height - shift_y_int)
    src_x_start = max(0, -shift_x_int)
    src_x_end = min(width, width - shift_x_int)
    
    dst_y_start = max(0, shift_y_int)
    dst_y_end = min(height, height + shift_y_int)
    dst_x_start = max(0, shift_x_int)
    dst_x_end = min(width, width + shift_x_int)
    
    # Copy valid region
    src_height = src_y_end - src_y_start
    src_width = src_x_end - src_x_start
    dst_height = dst_y_end - dst_y_start
    dst_width = dst_x_end - dst_x_start
    
    # Ensure regions have same size
    copy_height = min(src_height, dst_height)
    copy_width = min(src_width, dst_width)
    
    if copy_height > 0 and copy_width > 0:
        shifted_img[dst_y_start:dst_y_start+copy_height, 
                   dst_x_start:dst_x_start+copy_width] = \
            img[src_y_start:src_y_start+copy_height, 
                src_x_start:src_x_start+copy_width]
    
    return shifted_img

# --- Gradient Calculation (Manual Implementation) ---

def manual_convolve2d_optimized(image, kernel):
    """Performs 2D convolution using NumPy vectorization for better performance."""
    k_h, k_w = kernel.shape
    img_h, img_w = image.shape
    out_h = img_h - k_h + 1
    out_w = img_w - k_w + 1
    
    # Create output array
    output = np.zeros((out_h, out_w), dtype=np.float64)
    
    # Use efficient NumPy broadcasting instead of loops
    # This replaces the double for-loop with vectorized operations
    for i in range(k_h):
        for j in range(k_w):
            output += image[i:i+out_h, j:j+out_w] * kernel[i, j]
    
    # Handle padding
    pad_h_before = (k_h - 1) // 2
    pad_h_after = k_h - 1 - pad_h_before
    pad_w_before = (k_w - 1) // 2
    pad_w_after = k_w - 1 - pad_w_before
    
    padded_output = np.pad(output,
                         ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)),
                         mode='constant', constant_values=0)
    
    if padded_output.shape != image.shape:
        padded_output = padded_output[:img_h, :img_w]
        
    return padded_output

def calculate_gradients(img1_norm, img2_norm):
    """Calculates spatial (fx, fy) and temporal (ft) gradients."""
    print("Calculating image gradients...")
    height, width = img1_norm.shape
    kernel_x = np.array([[-0.25, 0.25], [-0.25, 0.25]], dtype=np.float64)
    kernel_y = np.array([[-0.25, -0.25], [0.25, 0.25]], dtype=np.float64)
    kernel_t = np.array([[0.25, 0.25], [0.25, 0.25]], dtype=np.float64)
    fx = np.zeros_like(img1_norm, dtype=np.float64)
    fy = np.zeros_like(img1_norm, dtype=np.float64)
    ft = np.zeros_like(img1_norm, dtype=np.float64)
    fx_img1 = manual_convolve2d_optimized(img1_norm, kernel_x)
    fx_img2 = manual_convolve2d_optimized(img2_norm, kernel_x)
    fx = fx_img1 + fx_img2
    fy_img1 = manual_convolve2d_optimized(img1_norm, kernel_y)
    fy_img2 = manual_convolve2d_optimized(img2_norm, kernel_y)
    fy = fy_img1 + fy_img2
    ft_img1 = manual_convolve2d_optimized(img1_norm, -kernel_t)
    ft_img2 = manual_convolve2d_optimized(img2_norm, kernel_t)
    ft = ft_img1 + ft_img2
    print("Gradients calculated.")
    return fx, fy, ft

# --- Horn-Schunck Algorithm (NumPy Implementation) ---

def horn_schunck_optical_flow(img1, img2, iterations, alpha):
    """Optimized version of Horn-Schunck optical flow calculation."""
    print(f"Starting Horn-Schunck: iterations={iterations}, alpha(lambda)={alpha}")
    fx, fy, ft = calculate_gradients(img1, img2)
    height, width = img1.shape
    u = np.zeros((height, width), dtype=np.float64)
    v = np.zeros((height, width), dtype=np.float64)
    avg_kernel = np.array([[1/12, 1/6, 1/12], [1/6, 0, 1/6], [1/12, 1/6, 1/12]], dtype=np.float64)
    
    # Precompute these values - they don't change in the iteration loop
    fx2 = fx * fx
    fy2 = fy * fy
    denominator_constant = alpha * alpha
    denominator = denominator_constant + fx2 + fy2 + 1e-6  # Add epsilon directly
    
    for i in range(iterations):
        if (i + 1) % 10 == 0 or i == 0: print(f"  Iteration {i+1}/{iterations}")
        
        # Use the optimized convolution function
        u_avg = manual_convolve2d_optimized(u, avg_kernel)
        v_avg = manual_convolve2d_optimized(v, avg_kernel)
        
        P = fx * u_avg + fy * v_avg + ft
        common_update_term = P / denominator
        
        u = u_avg - fx * common_update_term
        v = v_avg - fy * common_update_term
        
    print("Horn-Schunck calculation finished.")
    return u, v

# --- Visualization (Manual Implementation) ---

def draw_line_bresenham(image, x0, y0, x1, y1, color, thickness):
    """Draws a line using Bresenham's algorithm with thickness."""
    height, width = image.shape[:2]
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0

    while True:
        # Draw the pixel and its neighbors for thickness
        # thickness=0 means only the center pixel (1px line)
        # thickness=1 means center + 8 neighbors (3px line)
        for i in range(-thickness, thickness + 1):
            for j in range(-thickness, thickness + 1):
                plot_x, plot_y = x + i, y + j
                if 0 <= plot_y < height and 0 <= plot_x < width:
                    image[plot_y, plot_x] = color

        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            if x == x1: break
            err += dy
            x += sx
        if e2 <= dx:
            if y == y1: break
            err += dx
            y += sy

def draw_flow_arrow(image, x0, y0, x1, y1, color, thickness):
    """Draws a flow vector with a simple arrow head."""
    # Draw the main line
    draw_line_bresenham(image, x0, y0, x1, y1, color, thickness)
    
    # Simple arrow head (cross line at the end)
    # Calculate perpendicular direction for arrow head
    if x1 != x0 or y1 != y0:  # Avoid division by zero
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx*dx + dy*dy)
        if length > 0:
            # Normalize
            dx /= length
            dy /= length
            
            # Perpendicular direction
            px = -dy * 3
            py = dx * 3
            
            # Draw the cross line at the end point
            ax1 = int(x1 + px)
            ay1 = int(y1 + py)
            ax2 = int(x1 - px)
            ay2 = int(y1 - py)
            
            draw_line_bresenham(image, x1, y1, ax1, ay1, color, thickness)
            draw_line_bresenham(image, x1, y1, ax2, ay2, color, thickness)

def draw_optical_flow_vectors(image, u, v, full_save_path, scale=10, step=15, color=(0,0,255), thickness=0):
    """
    Draws flow vectors with arrows onto the image using Bresenham lines and saves it.
    """
    print(f"Drawing flow vectors (scale={scale}, step={step}, thickness={thickness*2+1}px)...")
    height, width = image.shape[:2]

    if image.ndim == 2:
        img_display = np.clip(image, 0, 255).astype(np.uint8)
        color_image = np.stack([img_display] * 3, axis=-1)
    elif image.dtype != np.uint8:
        color_image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        color_image = image.copy()

    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            try:
                flow_u = u[y, x]
                flow_v = v[y, x]
                
                # Only draw arrows with significant motion
                if abs(flow_u) > 0.01 or abs(flow_v) > 0.01:  
                    x_start, y_start = x, y
                    x_end = int(x_start + flow_u * scale)
                    y_end = int(y_start + flow_v * scale)
                    draw_flow_arrow(color_image, x_start, y_start, x_end, y_end, color, thickness)
            except IndexError:
                continue

    # Save the individual image
    save_image(color_image, full_save_path)

# --- Main Execution Logic ---

def main():
    """
    Main function: Loads image, calculates flow for parameter sets,
    saves individual results, and creates a composite table image using matplotlib.
    """
    start_time = time.perf_counter()  # <<< START TIMING >>>
    print("--- Starting Optical Flow Script (Optimized) ---")
    print(f"Script Base Directory: {SCRIPT_DIR}")
    print(f"Input Image Path: {INPUT_IMAGE_PATH}")
    print(f"Output Folder Path: {OUTPUT_FOLDER_PATH}")

    if not os.path.exists(OUTPUT_FOLDER_PATH):
        try:
            os.makedirs(OUTPUT_FOLDER_PATH)
            print(f"Created output directory: {OUTPUT_FOLDER_PATH}")
        except OSError as e:
            print(f"Error creating output directory '{OUTPUT_FOLDER_PATH}': {e}")
            sys.exit(1)

    img1_original = load_image_grayscale(INPUT_IMAGE_PATH)
    if img1_original is None:
        print("Exiting due to image loading failure.")
        sys.exit(1)

    grayscale_save_path = os.path.join(OUTPUT_FOLDER_PATH, "lena_grayscale.png")
    save_image(img1_original, grayscale_save_path)

    img2_shifted = shift_image_optimized(img1_original, SHIFT_AMOUNT, SHIFT_AMOUNT)
    shifted_save_path = os.path.join(OUTPUT_FOLDER_PATH, "lena_shifted.png")
    save_image(img2_shifted, shifted_save_path)

    results_paths = {}

    print("\n--- Processing Parameter Sets ---")
    for i, iterations in enumerate(ITERATION_SET):
        for j, alpha_lambda in enumerate(LAMBDA_SET):
            print(f"\nProcessing: Iterations={iterations}, Lambda(alpha)={alpha_lambda} (Row {j}, Col {i})")
            u, v = horn_schunck_optical_flow(img1_original, img2_shifted,
                                         iterations=iterations, alpha=alpha_lambda)

            alpha_str = str(alpha_lambda).replace('.', 'p')
            output_filename = f"HornSchunck_py_iter{iterations}_lamb{alpha_str}.png"
            full_output_path = os.path.join(OUTPUT_FOLDER_PATH, output_filename)

            # Draw flow vectors with arrow heads and save individual image
            draw_optical_flow_vectors(img2_shifted, u, v,
                                  full_save_path=full_output_path,
                                  scale=FLOW_VIS_SCALE, step=FLOW_VIS_STEP,
                                  color=FLOW_LINE_COLOR, thickness=FLOW_LINE_THICKNESS)

            if os.path.exists(full_output_path):
                 results_paths[(iterations, alpha_lambda)] = full_output_path
            else:
                 print(f"Warning: Failed to save or find result image for iter={iterations}, lambda={alpha_lambda}")

    print("\n--- Assembling Results Table using Matplotlib ---")
    if not results_paths or len(results_paths) != TABLE_ROWS * TABLE_COLS:
         print("Error: Not enough result images generated to create the table.")
         print(f"Expected {TABLE_ROWS * TABLE_COLS}, got {len(results_paths)}")
         sys.exit(1)

    # --- Create Table using Matplotlib Subplots ---
    # Adjust figsize based on image dimensions for better display
    # Load one image to get dimensions
    first_key = (ITERATION_SET[0], LAMBDA_SET[0])
    if first_key not in results_paths:
        print(f"Error: Cannot find result for {first_key} to determine image size.")
        sys.exit(1)
    first_img = load_result_image(results_paths[first_key])
    if first_img is None:
        print("Error: Could not load the first result image to determine dimensions.")
        sys.exit(1)
    img_height, img_width = first_img.shape[:2]

    # Estimate appropriate figure size (e.g., inches)
    # Assume ~100 DPI for screen, adjust scale factor as needed
    dpi = 50
    fig_width_in = (TABLE_COLS * img_width) / dpi * 1.1  # Add some padding factor
    fig_height_in = (TABLE_ROWS * img_height) / dpi * 1.2  # Add more vertical for titles

    fig, axs = plt.subplots(TABLE_ROWS, TABLE_COLS, figsize=(fig_width_in, fig_height_in),
                           sharex=True, sharey=True)  # Share axes might not be needed if ticks are off
    fig.suptitle('Horn-Schunck Optical Flow Results', fontsize=72)

    # Flatten axs array for easier iteration if needed, or use 2D indexing
    axs = axs.ravel()  # Flatten the 2D array of axes into 1D

    plot_index = 0
    for j, alpha_lambda in enumerate(LAMBDA_SET):  # Rows
        for i, iterations in enumerate(ITERATION_SET):  # Columns
            key = (iterations, alpha_lambda)
            ax = axs[plot_index]  # Get the current subplot axes

            if key in results_paths:
                img_path = results_paths[key]
                result_img = load_result_image(img_path)
                if result_img is not None:
                    ax.imshow(result_img)
                else:
                    ax.text(0.5, 0.5, 'Load Failed', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    print(f"Failed to load image for iter={iterations}, lambda={alpha_lambda}")
            else:
                ax.text(0.5, 0.5, 'Missing', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                print(f"Missing result for iter={iterations}, lambda={alpha_lambda}")

            # Add titles/labels
            if j == 0:  # Top row
                ax.set_title(f'Iterations = {iterations}',fontsize = 48)
            if i == 0:  # Leftmost column
                ax.set_ylabel(f'Lambda = {alpha_lambda}', rotation=90, size = 42)

            # Hide ticks and labels on the image axes
            ax.set_xticks([])
            ax.set_yticks([])

            plot_index += 1

    # Adjust layout to prevent labels overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust rect to make space for suptitle

    # Save the final composite figure
    composite_save_path = os.path.join(OUTPUT_FOLDER_PATH, COMPOSITE_IMAGE_NAME)
    try:
        fig.savefig(composite_save_path, dpi=dpi)  # Save the figure with specified DPI
        print(f"\nSuccessfully saved composite results table to: {composite_save_path}")
    except Exception as e:
        print(f"\nFailed to save composite results table: {e}")
    end_time = time.perf_counter()  # <<< END TIMING >>>
   
    total_time = end_time - start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60

    # Print the execution time in the desired format
    print(f"\n--- Total Execution Time: {minutes} minutes {seconds:.2f} seconds ({total_time:.2f} total seconds) ---")  # <<< PRINT TIME >>>
    print("--- Optical Flow Script Finished ---")

if __name__ == "__main__":
    main()