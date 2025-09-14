import numpy as np
import cv2
from src.datasets.dataloader import ProcessNewData, GetData
from glob import glob
from pathlib import Path


def create_video_from_image_paths(image_paths, output_path="output.mp4", fps=30):
    """
    Create a video from a list of image file paths.

    Parameters:
    image_paths: list of strings or Path objects pointing to image files
    output_path: string, path where the video will be saved
    fps: int, frames per second for the output video
    """
    # Read the first image to get dimensions
    first_image = cv2.imread(str(image_paths[0]))
    if first_image is None:
        raise ValueError(f"Could not read image: {image_paths[0]}")

    height, width = first_image.shape[:2]

    # Create video writer object
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        # If CUDA is available, use hardware acceleration
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
    else:
        # Fall back to standard H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each image
    for i, img_path in enumerate(image_paths):
        # Read image
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"Warning: Could not read image {img_path}, skipping...")
            continue

        # Ensure image has the same dimensions as the first image
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))

        # Write to video
        out.write(img)

        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(image_paths)} images")

    # Release the video writer
    out.release()
    print(f"\nVideo saved to {output_path}")


def create_video_from_matrix(matrix, output_path="output.mp4", fps=30):
    """
    Convert a 3D numpy array of matrix into an MP4 video file.

    Parameters:
    matrix: numpy array of shape (n_matrix, height, width)
    output_path: string, path where the video will be saved
    fps: int, matrix per second for the output video
    """
    # Ensure matrix are in uint8 format (0-255)
    if matrix.dtype != np.uint8:
        # Normalize to 0-255 if not already in that range
        matrix = (
            (matrix - matrix.min()) * (255.0 / (matrix.max() - matrix.min()))
        ).astype(np.uint8)

    # Get dimensions
    n_matrix, height, width = matrix.shape

    # Create video writer object
    # For MP4, we use H.264 codec
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        # If CUDA is available, use hardware acceleration
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
    else:
        # Fall back to standard H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    # Write each frame
    for i in range(n_matrix):
        frame = matrix[i]
        out.write(frame)

    # Release the video writer
    out.release()
    print(f"Video saved to {output_path}")


def video_from_mds():
    test_data = GetData(type_data="no_type")

    datasets = test_data.datasets
    labels = test_data.labels
    paths = test_data.paths

    labels = np.argmax(labels, axis=1)

    real_label = ["pms", "bms", "cms"]
    counters = [0, 0, 0]
    for i, (label, dataset) in enumerate(zip(labels, datasets)):
        create_video_from_matrix(
            dataset.transpose(2, 0, 1),
            output_path=f"results/statistics/videos/mds/mds_{real_label[label]}_{paths[i].split('/')[-2]}.mp4",
            fps=30,
        )
        counters[label] += 1


def video_from_images():
    classes = ["2019_04_09_pms1000", "2019_04_09_bms1000", "2019_04_09_cms1000"]
    test_data = GetData(type_data="no_type")

    real_label = ["pms", "bms", "cms"]
    datasets = test_data.datasets
    labels = test_data.labels
    paths = test_data.paths

    labels = np.argmax(labels, axis=1)

    image_names = [(path.split("/")[-4], path.split("/")[-2]) for path in paths]

    counters = [0, 0, 0]
    for i, image_name in enumerate(image_names):
        image_paths = list(
            np.sort(glob(f"./src/datasets/Automotive/{image_name[0]}/images_0/*"))
        )
        current_path = f"./src/datasets/Automotive/{image_name[0]}/images_0/0000{image_name[1]}.jpg"
        index = image_paths.index(current_path)
        frames_image_paths = image_paths[index - 15 : index + 1]
        create_video_from_image_paths(
            image_paths=frames_image_paths,
            output_path=f"results/statistics/videos/frames/frames_{real_label[labels[i]]}_{image_name[1]}.mp4",
            fps=10,
        )
        counters[labels[i]] += 1


if __name__ == "__main__":
    # Assuming your matrix is called 'matrix'
    # Generate sample data

    # video_from_mds()
    video_from_images()
