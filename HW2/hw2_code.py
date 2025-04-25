import cv2
import os
import math

# Constants
FOCAL_LENGTH_COUNT = 3
DISPLACEMENT_COUNT = 5
DISTANCE_COUNT = 3
CAMERA_SENSOR_WIDTH = 23.4

def create_image_slices(source_image, reference_position, slice_height, slice_width):
    """Generate image slices for comparison"""
    slice_collection = []
    for horizontal_pos in range(reference_position[0], source_image.shape[1] - slice_width, 1):
        current_slice = source_image[reference_position[1]:reference_position[1] + slice_height, horizontal_pos:horizontal_pos + slice_width].copy()
        slice_collection.append(((horizontal_pos, reference_position[1]), current_slice))
    return slice_collection

def find_horizontal_shift(reference_patch, reference_position, target_image):
    """Find horizontal shift of reference patch in target image"""
    slice_collection = create_image_slices(target_image, reference_position, reference_patch.shape[0], reference_patch.shape[1])
    best_error = float('inf')
    best_match_location = (0, 0)
    max_search_distance = 600.0

    for location, current_slice in slice_collection:
        horizontal_distance = location[0] - reference_position[0]
        # Only consider slices within the search range
        if horizontal_distance <= max_search_distance:
            # Compare reference patch with current slice
            current_error = compute_image_difference(reference_patch, current_slice)
            if current_error < best_error:
                best_error = current_error
                best_match_location = location

    # Return only the horizontal shift
    return best_match_location[0] - reference_position[0]

def compute_image_difference(first_image, second_image):
    """Compute total difference between two images"""
    # Check image dimensions
    if len(first_image.shape) != len(second_image.shape):
        raise ValueError("Images must have the same number of channels.")

    if first_image.shape[:2] != second_image.shape[:2]:
        raise ValueError("Images must have the same height and width.")

    image_height, image_width = first_image.shape[:2]
    channel_count = first_image.shape[2] if len(first_image.shape) > 2 else 1

    difference_sum = 0

    for row in range(image_height):
        for col in range(image_width):
            if channel_count > 1:
                for channel in range(channel_count):
                    # Color images (multi-channel)
                    val1 = first_image[row][col][channel]
                    val2 = second_image[row][col][channel]
                    diff = abs(int(val1) - int(val2)) #make sure they are integers before subtraction.
                    difference_sum += diff

    return difference_sum

def calculate_theoretical_field_of_view(focal_length):
    """Calculate theoretical field of view based on focal length"""
    return 2 * math.atan(CAMERA_SENSOR_WIDTH / 2 / focal_length) * 180 / math.pi

def calculate_millimeters_per_pixel(actual_displacement, pixel_displacement):
    """Calculate millimeters per pixel conversion factor"""
    return float(actual_displacement) / pixel_displacement

def calculate_measured_field_of_view(image_width, object_distance, mm_per_pixel):
    """Calculate measured field of view based on image dimensions and distance"""
    return 2 * math.atan(image_width * mm_per_pixel / 2 / object_distance) * 180 / math.pi

def main():
    # Configuration parameters
    focal_lengths = [18, 53, 135]
    displacements = [0, 1, 5, 10, 20]
    distances = [600, 1200, 1800]

    # Initialize data storage
    image_library = {}
    for focal in focal_lengths:
        image_library[focal] = {}
        for dist in distances:
            image_library[focal][dist] = {}
            for displ in displacements:
                image_library[focal][dist][displ] = None

    # Load all images
    print("Beginning image file retrieval...")
    for focal in focal_lengths:
        for dist in distances:
            for displ in displacements:
                # Check for images with different case extensions
                file_path = f"Photo/{focal}mm/{dist}mm_{displ}mm.jpg"
                with open(file_path, 'rb') as f:
                    f.seek(-2,2)
                    loaded_image = cv2.imread(file_path)
                    if f.read() != '\xff\xd9':
                        print(file_path)
                

                if loaded_image is None:
                    file_path = f"Photo/{focal}mm/{dist}mm_{displ}mm.JPG"
                    loaded_image = cv2.imread(file_path)

                if loaded_image is None:
                    print(f"Notice: Unable to locate image at {file_path}")
                    continue

                image_library[focal][dist][displ] = loaded_image.copy()

    # Initialize displacement measurement storage
    displacement_measurements = {}
    for focal in focal_lengths:
        displacement_measurements[focal] = {}
        for dist in distances:
            displacement_measurements[focal][dist] = {}
            for displ in displacements:
                displacement_measurements[focal][dist][displ] = 0

    # Process each configuration
    print("Initiating region selection for object tracking...")
    for focal in focal_lengths:
        for dist in distances:
            # Skip missing configurations
            if image_library[focal][dist][0] is None:
                print(f"Bypassing {focal}mm_{dist}mm - Required initial image is absent")
                continue

            config_name = f"{focal}mm_{dist}mm"
            selection_window = f"Draw an RoI rectangle for {config_name}"
            cv2.namedWindow(selection_window, cv2.WINDOW_NORMAL)

            # Have user select tracking region
            region = cv2.selectROI(selection_window, image_library[focal][dist][0])
            cv2.destroyWindow(selection_window)

            # Extract the selected region
            reference_patch = image_library[focal][dist][0][
                int(region[1]):int(region[1] + region[3]),
                int(region[0]):int(region[0] + region[2])
            ].copy()
            reference_position = (int(region[0]), int(region[1]))

            # Calculate displacement for each test image
            for displ in displacements:
                if image_library[focal][dist][displ] is not None:
                    displacement_measurements[focal][dist][displ] = find_horizontal_shift(
                        reference_patch, reference_position, image_library[focal][dist][displ]
                    )

    # Generate results file
    print("Commencing output file creation...")
    with open("output/output.csv", "w") as results_file:
        results_file.write("Focal length(mm), Object distance(mm), Object actual displacement(mm), Object displacement(pixel), mm/pixel, FOV theoretical value(degree), FOV measured value(degree)\n")

        for focal in focal_lengths:
            for dist in distances:
                for displ in displacements:
                    # Skip missing or invalid measurements
                    if image_library[focal][dist][displ] is None:
                        continue
                    
                    # Calculate metrics
                    mm_per_pixel_ratio = 0
                    measured_fov = 0
                    if displ != 0 and displacement_measurements[focal][dist][displ] != 0:
                        mm_per_pixel_ratio = calculate_millimeters_per_pixel(displ, displacement_measurements[focal][dist][displ])
                        image_width = image_library[focal][dist][displ].shape[1]
                        measured_fov = calculate_measured_field_of_view(image_width, dist, mm_per_pixel_ratio)
                        
                    theoretical_fov = calculate_theoretical_field_of_view(focal)

                    # Write row to CSV

                    results_file.write(f"{focal},{dist},{displ},{displacement_measurements[focal][dist][displ]},{mm_per_pixel_ratio},{theoretical_fov},{measured_fov}\n")

    print("Analysis complete. Results saved to output/output.csv")

if __name__ == "__main__":
    main()