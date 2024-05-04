import os
import cv2

def apply_clahe_to_folder(input_folder, output_folder, clip_limit=2.0, grid_size=(8, 8)):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through subfolders (classes)
    for class_folder in os.listdir(input_folder):
        class_folder_path = os.path.join(input_folder, class_folder)
        output_class_folder_path = os.path.join(output_folder, class_folder)

        # Create output class folder if it doesn't exist
        if not os.path.exists(output_class_folder_path):
            os.makedirs(output_class_folder_path)

        # Iterate through images in the class folder
        for filename in os.listdir(class_folder_path):
            input_image_path = os.path.join(class_folder_path, filename)
            output_image_path = os.path.join(output_class_folder_path, filename)

            # Read the image
            image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            clahe_image = clahe.apply(image)

            # Save the processed image
            cv2.imwrite(output_image_path, clahe_image)

if __name__ == "__main__":
    # Specify your dataset folder
    main_dataset_folder = "val/"

    # Specify the output folder for processed images
    output_folder = "val_clahe"

    # Set CLAHE parameters
    clip_limit = 2.0
    grid_size = (8, 8)

    # Apply CLAHE to the dataset
    apply_clahe_to_folder(main_dataset_folder, output_folder, clip_limit, grid_size)
