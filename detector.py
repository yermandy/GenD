import argparse
import heapq
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from src.retinaface import RetinaFace, prepare_model


def max_spread_permutation_pq(N, start=0):
    """
    Generate a permutation of 0..N-1 such that at each step
    the next element is the one whose minimum distance to
    all previously chosen elements is maximized, using a
    priority queue to speed up selection.

    Args:
        N (int): Length of the permutation.
        start (int): The first element in the permutation (default 0).

    Returns:
        List[int]: A list representing the permutation.
    """
    if not (0 <= start < N):
        raise ValueError("`start` must be in the range [0, N-1]")

    # Initialize chosen list and distance map
    chosen = [start]
    dist = {i: abs(i - start) for i in range(N) if i != start}

    # Build a max-heap (use negative distances for heapq)
    heap = [(-d, i) for i, d in dist.items()]
    heapq.heapify(heap)

    # Greedily pick elements
    while heap:
        # Pop until we find a valid (up-to-date) entry
        while True:
            neg_d, candidate = heapq.heappop(heap)
            current = -neg_d
            # Only accept if it matches the latest dist
            if dist.get(candidate, -1) == current:
                break

        # Add the selected candidate
        chosen.append(candidate)
        # Remove it from dist-map
        del dist[candidate]

        # Update distances for remaining elements
        for other in list(dist.keys()):
            new_d = abs(other - candidate)
            if new_d < dist[other]:
                dist[other] = new_d
                heapq.heappush(heap, (-new_d, other))

    return chosen


def get_video_frames_generator(
    source_path: str,
    mask_path: str,
    stride: int = 1,
    num_frames=32,
    mode="at_least",
):
    video = cv2.VideoCapture(source_path)
    if not video.isOpened():
        print(f"Warning: Video {source_path} cannot be opened!")
        return

    video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if mask_path is not None:
        mask_video = cv2.VideoCapture(mask_path)

        if not mask_video.isOpened():
            print(f"Warning: Mask video {mask_path} cannot be opened!")
            return

        mask_frames = int(mask_video.get(cv2.CAP_PROP_FRAME_COUNT))

        if video_frames != mask_frames:
            print(
                f"Warning: {source_path} and {mask_path} have different number of frames {video_frames} vs {mask_frames}!"
            )

        total_frames = min(video_frames, mask_frames)
    else:
        mask_video = None
        total_frames = video_frames

    if not video.isOpened():
        raise Exception(f"Could not open video at {source_path}")

    # Get the mode
    if mode == "fixed_num_frames":
        # Get the frame rate of the video by dividing the number of frames by the duration (same interval between frames)
        frame_ids = np.linspace(0, total_frames - 1, num_frames, endpoint=True, dtype=int)
    elif mode == "fixed_stride":
        # Get the frame rate of the video by dividing the number of frames by the duration (same interval between frames)
        frame_ids = np.arange(0, total_frames, stride, dtype=int)
    elif mode == "at_least":
        frame_ids = max_spread_permutation_pq(total_frames, start=total_frames // 2)
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'fixed_num_frames', 'fixed_stride', or 'at_least'.")

    # Iterate through the selected frame IDs
    for frame_id in frame_ids:
        # Set the video capture position to the desired frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, frame = video.read()

        if mask_video is not None:
            mask_video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            success_mask, mask = mask_video.read()
            if not success_mask:
                print(f"Warning: Failed to read mask frame {frame_id} of {mask_path}. Skipping.")
                continue

            yield frame, frame_id, mask

        else:
            # Check if the frame was successfully read
            if not success:
                print(f"Warning: Failed to read frame {frame_id} of {source_path}. Skipping.")
                continue

            yield frame, frame_id, None

    # Release the video capture object
    video.release()

    if mask_video is not None:
        mask_video.release()


def align_face(
    img: np.ndarray,
    landmarks: np.ndarray,
    target_size: None | tuple = None,
    scale: float = 1.3,
    mask: np.ndarray = None,
):
    """
    Aligns a face based on 5-point facial landmarks (eyes, nose, mouth corners).

    Args:
        img: Input image containing the face
        landmarks: 5-point facial landmarks array with shape (5, 2)
        target_size: Desired output size as (width, height) tuple
        scale: Scaling factor to control how much context around the face to include
        stabilize_features: Whether to use standard reference points for consistent alignment
        return_transform: Whether to return the transformation matrix
        mask: Resize mask the same way as img

    Returns:
        Aligned face image with specified target_size
        Optionally returns the transformation matrix if return_transform=True
    """
    dst = np.array(
        [
            [0.34, 0.46],
            [0.66, 0.46],
            [0.5, 0.64],
            [0.37, 0.82],
            [0.63, 0.82],
        ],
        dtype=np.float32,
    )

    if target_size is None:
        # Compute desired distances between all pairs
        desired_dists = np.linalg.norm(landmarks[:, None, :] - landmarks[None, :, :], axis=-1)

        # Destination distances between all pairs
        dst_dists = np.linalg.norm(dst[:, None, :] - dst[None, :, :], axis=-1)

        # Take upper triangle of the distance matrix
        upper_triangle_indices = np.triu_indices(len(dst), k=1)
        dst_dists = dst_dists[upper_triangle_indices]
        desired_dists = desired_dists[upper_triangle_indices]

        # Approximate target size
        approx_size = np.round(np.mean(desired_dists / dst_dists) * scale).astype(int)
        target_size = (approx_size, approx_size)

    dst[:, 0] = dst[:, 0] * target_size[0]
    dst[:, 1] = dst[:, 1] * target_size[1]

    margin_rate = scale - 1
    x_margin = target_size[0] * margin_rate / 2.0
    y_margin = target_size[1] * margin_rate / 2.0

    # move
    dst[:, 0] += x_margin
    dst[:, 1] += y_margin

    # resize
    dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
    dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

    src = landmarks.astype(np.float32)

    M = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]

    img = cv2.warpAffine(img, M, target_size, flags=cv2.INTER_LINEAR)

    # Warp landmarks, show
    # landmarks = cv2.transform(np.expand_dims(landmarks, axis=0), M)[0]
    # for point in landmarks:
    #     cv2.circle(img, tuple(point.astype(int)), 2, (0, 255, 0), -1)

    if mask is not None:
        mask = cv2.warpAffine(mask, M, target_size, flags=cv2.INTER_NEAREST)

    return img, mask


def process_video(
    source_path,
    target_path,
    mask_path,
    model: RetinaFace,
    scale=1.3,
    target_size=(256, 256),
    stride=1,
    num_frames=32,
    mode="at_least",
    skip_processed_videos=False,
    skip_processed_frames=False,
):
    frame_save_path = target_path.replace(".mp4", "/frames")

    # Skip if frame_save_path exists
    if skip_processed_videos and os.path.exists(frame_save_path):
        print(f"Frames for {source_path} already processed.")
        return
    else:
        print(f"Processing {source_path}")

    # Create a frame generator from video path for iteration of frames
    frame_generator = get_video_frames_generator(
        source_path,
        mask_path,
        stride=stride,
        num_frames=num_frames,
        mode=mode,
    )
    # desc = f"Processing {os.path.basename(source_path)}"

    num_saved = 0
    for frame, frame_id, mask in frame_generator:
        frame_filename = os.path.join(frame_save_path, f"frame_{frame_id:04d}.png")

        if skip_processed_frames and os.path.exists(frame_filename):
            print(f"Frame {frame_id} of {source_path} already processed.")
            num_saved += 1
            if mode in ["fixed_stride", "at_least"] and num_saved >= num_frames and num_frames != -1:
                break
            continue

        try:
            preds = model.detect(frame)
        except Exception as e:
            print(f"Error during detection: {e}")
            continue

        xyxy, landmarks = preds

        if len(xyxy) == 0:
            print(f"No faces detected in frame {frame_id} of {source_path}")
            continue

        selected_landmarks = None

        if mask is not None:
            # It is possible that the mask is empty -> skip this frame
            if mask.sum() == 0:
                print(f"Warning: Mask is empty for frame {frame_id} of {source_path}. Skipping.")
                continue

            # Convert mask to grayscale if it's not already
            mask_img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask

            # Threshold the mask to create a binary mask
            mask_img = cv2.threshold(mask_img, 1, 255, cv2.THRESH_BINARY)[1]

            # Find the face that intersects the most with the mask
            best_landmarks = None
            max_intersection = 0
            for i in range(len(xyxy)):
                # Get the bounding box coordinates
                x1, y1, x2, y2 = xyxy[i, :4].astype(int)

                # Create a mask for the face
                face_mask = np.zeros_like(mask_img)
                face_mask[y1:y2, x1:x2] = 255

                # Calculate the intersection between the face mask and the provided mask
                intersection = np.sum(np.logical_and(face_mask, mask_img))

                # Update the best face if the intersection is greater than the current maximum
                if intersection > max_intersection:
                    max_intersection = intersection
                    best_landmarks = landmarks[i]

            # If a face was found, use it; otherwise, skip this frame
            if best_landmarks is not None:
                selected_landmarks = best_landmarks
            else:
                print(f"No suitable face found in frame {frame_id} of {source_path} with the provided mask.")
                continue

        # """
        # Select landmarks of the largest face if not using mask
        if selected_landmarks is None:
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            idx = np.argmax(areas)
            selected_landmarks = landmarks[idx]

        # Show all landmarks
        # for L, B in zip(landmarks, xyxy):
        #     for point in L:
        #         cv2.circle(frame, tuple(point.astype(int)), 2, (0, 255, 0), -1)
        #     cv2.rectangle(frame, tuple(B[0:2].astype(int)), tuple(B[2:4].astype(int)), (0, 255, 0), 2)

        # Align the face
        aligned_face, _ = align_face(frame, selected_landmarks, target_size=target_size, scale=scale)

        # Save the aligned face
        os.makedirs(frame_save_path, exist_ok=True)
        cv2.imwrite(frame_filename, aligned_face)
        # """

        num_saved += 1

        if mode in ["fixed_stride", "at_least"] and num_saved >= num_frames and num_frames != -1:
            break

    if num_saved == 0:
        print(f"No faces were saved from {source_path}. Check the detection threshold or input video.")

    return frame_save_path


def process_image(
    source_path,
    target_path,
    model: RetinaFace,
    scale=1.3,
    target_size=(256, 256),
    skip_processed_frames=False,
):
    """Processes a single image file."""
    if skip_processed_frames and os.path.exists(target_path):
        print(f"Image {source_path} already processed.")
        return target_path
    else:
        print(f"Processing {source_path}")

    img = cv2.imread(source_path)
    if img is None:
        print(f"Failed to read image {source_path}")
        return None

    try:
        preds = model.detect(img)
    except Exception as e:
        print(f"Error during detection: {e}")
        return None

    xyxy, landmarks = preds

    if len(xyxy) == 0:
        print(f"No faces detected in {source_path}")
        return None

    # Select landmarks of the largest face
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    idx = np.argmax(areas)
    landmarks = landmarks[idx]

    # Align the face
    aligned_face, _ = align_face(img, landmarks, target_size=target_size, scale=scale)

    # Save the aligned face
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    cv2.imwrite(target_path, aligned_face)
    return target_path


def get_output_path(source_path, input_folder, output_folder):
    # Example: source_path = input_folder + new_source_path``
    new_source_path = source_path.replace(input_folder, os.path.basename(input_folder))
    # Create directory for each video
    new_source_path = new_source_path.replace(".mp4", "")
    # Place it in the output folder
    output_path = os.path.join(output_folder, new_source_path)
    return output_path


def get_mask_path(input_folder, input_mask_folder, source_path):
    if input_mask_folder is not None:
        # Change the input folder to the mask folder
        source_path = source_path.replace(input_folder, input_mask_folder)

        #! FF++ has masks named the same way as original videos
        if "FaceForensics" in source_path or "FF++" in source_path:
            return source_path

        #! Else assume masks are named with _mask suffix
        source_path = source_path.replace(".mp4", "_mask.mp4")
        return source_path
    return None


def process_mixed_types(
    input_folder_or_file: str | list[str],
    input_mask_folder: None | str,
    model: RetinaFace,
    num_workers=1,
    scale=1.3,
    target_size=(256, 256),
    stride=1,
    num_frames=32,
    mode: str = "fixed_num_frames",
    output_folder: str = "outputs",
    possible_extensions: tuple[str] = ("mp4", "jpg", "png", "jpeg"),
    skip_processed_videos: bool = False,
    skip_processed_frames: bool = False,
):
    if os.path.isfile(input_folder_or_file):
        # If input is a file
        if input_folder_or_file.endswith(possible_extensions):
            # If input is a media file
            files = [input_folder_or_file]
        elif input_folder_or_file.endswith("txt"):
            # If input is a txt file
            with open(input_folder_or_file, "r") as f:
                files = f.read().splitlines()

    else:
        # If input is a folder
        files = find_files(input_folder_or_file, possible_extensions)

    if not files:
        print(f"No files found in {input_folder_or_file}")
        return

    def process(source_path):
        output_path = get_output_path(source_path, input_folder_or_file, output_folder)

        if source_path.endswith(".mp4"):
            mask_path = get_mask_path(input_folder_or_file, input_mask_folder, source_path)
            try:
                return process_video(
                    source_path,
                    output_path,
                    mask_path,
                    model,
                    scale=scale,
                    target_size=target_size,
                    stride=stride,
                    num_frames=num_frames,
                    mode=mode,
                    skip_processed_videos=skip_processed_videos,
                    skip_processed_frames=skip_processed_frames,
                )
            except Exception as e:
                print(f"Error processing video {source_path}: {e}")
        else:
            try:
                return process_image(
                    source_path,
                    output_path,
                    model,
                    scale=scale,
                    target_size=target_size,
                    skip_processed_frames=skip_processed_frames,
                )
            except Exception as e:
                print(f"Error processing image {source_path}: {e}")

    files = sorted(files)  # Sort files for consistent processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process, file) for file in files]
        for future in tqdm(futures, desc=f"Processing videos in {input_folder_or_file}", leave=True):
            future.result()

    print("Processing complete.")


def find_files_fd(start_dir, extensions):
    """
    Finds files with given extensions recursively using the 'fd' command-line tool.

    Args:
        start_dir (str): The directory to start searching from.
        extensions (list): A list of file extensions without the leading dot (e.g., ['png', 'jpg']).

    Returns:
        list: A list of full path strings for each found file. Returns empty list if fd fails.

    Raises:
        FileNotFoundError: If the 'fd' command is not found in the system's PATH.
    """
    if not os.path.isdir(start_dir):
        print(f"Error: Start directory not found: {start_dir}")
        return []

    try:
        # Build the command. Use -e for each extension.
        command = ["fd", "--type", "f", "--type", "l"]  # Find only files or links to files
        for ext in extensions:
            # fd expects extensions without the dot
            command.extend(["--extension", ext])
        # Add the pattern ('.' matches everything, filtering is done by extension)
        # and the directory to search
        command.extend([".", start_dir])

        # Run the command
        result = subprocess.run(
            command,
            capture_output=True,  # Capture stdout and stderr
            text=True,  # Decode output as text (UTF-8 by default)
            check=False,  # Do not raise exception on non-zero exit code automatically
            encoding="utf-8",  # Be explicit about encoding
        )

        # Check if fd ran successfully
        if result.returncode != 0:
            # fd returns specific exit codes, e.g., 1 for errors, 2 if pattern not found (but we use '.')
            # We mainly care if the command executed but maybe found nothing or had an issue.
            # Check stderr for actual errors.
            if result.stderr:
                print(f"Error running fd (code {result.returncode}): {result.stderr.strip()}")
            # If stderr is empty but code isn't 0, it might just mean no files found, which is okay.
            # We return an empty list in case of errors or no files found.
            return []  # Return empty list on error or if no files found

        # fd outputs one path per line. Split the output.
        # .strip() removes potential leading/trailing whitespace/newlines
        file_list = result.stdout.strip().splitlines()
        return file_list

    except FileNotFoundError:
        raise  # Re-raise the exception so the caller knows fd is missing

    except Exception as e:
        print(f"An unexpected error occurred while running fd: {e}")
        return []  # Return empty list on other unexpected errors


def find_files_glob(start_dir, extensions):
    """
    Finds files with given extensions recursively using glob.

    Args:
        start_dir (str): The directory to start searching from.
        extensions (list): A list of file extensions without the leading dot (e.g., ['png', 'jpg']).

    Returns:
        list: A list of full path strings for each found file.
    """
    files = []
    for ext in extensions:
        files.extend(glob(f"{start_dir}/**/*{ext}", recursive=True))
    return sorted(f for f in files if os.path.isfile(f))


def find_files(start_dir, extensions):
    try:
        return find_files_fd(start_dir, extensions)
    except Exception:
        return find_files_glob(start_dir, extensions)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder_or_file",
        type=str,
        required=True,
        help="Path to the input folder containing videos or images.",
    )
    parser.add_argument(
        "--mask_folder",
        type=str,
        default=None,
        help="Path to the input folder containing masks (optional).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker threads.",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        default=1.3,
        help="Scale factor for face alignment.",
    )
    parser.add_argument(
        "--target_size",
        type=str,
        default="256,256",
        help="Target size for aligned faces as width, height (e.g., 256,256) or 'none'.",
    )
    parser.add_argument(
        "--det_thres",
        type=float,
        default=0.4,
        help="Detection threshold for RetinaFace.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="at_least",
        choices=["fixed_num_frames", "fixed_stride", "at_least"],
        help="Mode for frame extraction from videos ('fixed_num_frames', 'fixed_stride', or 'at_least').",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for frame extraction from videos (only used in 'fixed_stride' mode).",
    )
    parser.add_argument(
        "-n",
        "--num_frames",
        type=int,
        default=32,
        help="Maximum number of frames to extract from each video, -1 for all frames.",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="outputs",
        help="Output folder for the preprocessed images.",
    )
    parser.add_argument(
        "--skip_processed_videos",
        action="store_true",
        help="Skip videos that have already been processed.",
    )
    parser.add_argument(
        "--skip_processed_frames",
        action="store_true",
        help="Skip frames that have already been processed.",
    )
    args = parser.parse_args()
    args.target_size = parse_target_size(args.target_size)
    return args


def parse_target_size(target_size_str):
    try:
        width, height = map(int, target_size_str.split(","))
        return (width, height)
    except ValueError:
        if "none" in target_size_str.lower():
            return None
        raise ValueError("Invalid target_size format. Use 'width,height' or 'none'.")


def main():
    args = get_args()

    model = prepare_model(args.det_thres)

    process_mixed_types(
        input_folder_or_file=args.input_folder_or_file,
        input_mask_folder=args.mask_folder,
        model=model,
        num_workers=args.num_workers,
        scale=args.scale,
        target_size=args.target_size,
        stride=args.stride,
        num_frames=args.num_frames,
        mode=args.mode,
        output_folder=args.output_folder,
        skip_processed_videos=args.skip_processed_videos,
        skip_processed_frames=args.skip_processed_frames,
    )

    exit(0)


if __name__ == "__main__":
    main()
