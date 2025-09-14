import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from src.datasets.dataloader import GetData

import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def create_doppler_video(
    data,
    output_filename="doppler_video.avi",
    fps=10,
    time_max=0.45,
    velocity_range=(-8, 8),
):
    """
    Create a video from a sequence of micro-Doppler maps

    Parameters:
    data: numpy array of shape (frames, height, width)
    output_filename: string, name of output video file
    fps: frames per second for the output video
    time_max: maximum time value in seconds
    velocity_range: tuple of (min_velocity, max_velocity) in m/s
    """
    # Get dimensions
    n_frames, height, width = data.shape

    # Create time and velocity axes
    time = np.linspace(0, time_max, width)
    velocity = np.linspace(velocity_range[0], velocity_range[1], height)

    # Initialize video writer
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    # First frame to get dimensions
    fig = plt.figure(figsize=(10, 8))
    canvas = FigureCanvas(fig)
    plt.pcolormesh(time, velocity, data[0], shading="auto", cmap="jet", vmin=0, vmax=4)
    plt.colorbar()
    plt.xlabel("time (s)")
    plt.ylabel("velocity (m/s)")
    plt.title("micro-Doppler map")
    plt.xlim(0, time_max)
    plt.ylim(velocity_range)
    plt.grid(True, alpha=0.3)

    # Draw the plot to get the size
    canvas.draw()
    # Get the RGBA buffer from the figure
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    image = buf.reshape(h, w, 4)
    # Convert RGBA to RGB
    image = image[:, :, :3]

    height, width, _ = image.shape

    # Create video writer with the correct dimensions
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    plt.close()

    # Process each frame
    for i in range(n_frames):
        # Create figure for this frame
        fig = plt.figure(figsize=(10, 8))
        canvas = FigureCanvas(fig)

        # Plot the data
        plt.pcolormesh(
            time, velocity, data[i], shading="auto", cmap="jet", vmin=0, vmax=4
        )
        plt.colorbar()
        plt.xlabel("time (s)")
        plt.ylabel("velocity (m/s)")
        plt.title(f"micro-Doppler map")
        plt.xlim(0, time_max)
        plt.ylim(velocity_range)
        plt.grid(True, alpha=0.3)

        # Convert plot to image
        canvas.draw()
        # Get the RGBA buffer from the figure
        w, h = canvas.get_width_height()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        image = buf.reshape(h, w, 4)
        # Convert RGBA to RGB
        image = image[:, :, :3]

        # OpenCV uses BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Write frame
        out.write(image)

        # Clear the current figure
        plt.close()

        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{n_frames} frames")

    # Release the video writer
    out.release()
    print(f"Video saved as {output_filename}")


def video_from_mds():
    test_data = GetData(type_data="no_type")

    datasets = test_data.datasets
    labels = test_data.labels
    paths = test_data.paths

    labels = np.argmax(labels, axis=1)

    indexes = np.argsort(paths)

    datasets = [datasets[index] for index in indexes]
    labels = [labels[index] for index in indexes]
    paths = [paths[index] for index in indexes]

    real_label = ["pms", "bms", "cms"]
    # breakpoint()
    frames = []
    for i, (label, dataset) in enumerate(zip(labels, datasets)):
        if "bms" == real_label[label] and int(paths[i].split("/")[-2]) <= 160:
            print(f"mds_{real_label[label]}_{paths[i].split('/')[-2]}.mp4")
            frames.append(dataset.transpose(2, 0, 1))
        # create_doppler_video(dataset.transpose(2, 0, 1), f'mds_{real_label[label]}_{paths[i].split("/")[-2]}.mp4', fps=30)
        # counters[label] += 1
        # break
    frames = np.vstack(frames)
    create_doppler_video(frames, f"mds_cms.mp4", fps=120)


if __name__ == "__main__":
    video_from_mds()
