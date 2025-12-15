import os

# set GRADIO_TEMP_DIR
os.environ["GRADIO_TEMP_DIR"] = "./tmp/gradio"


import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import gradio as gr
import imageio
import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from PIL import Image

# Ensure project root on sys.path
try:
    import autorootcwd
except Exception:
    THIS = Path(__file__).resolve()
    ROOT = THIS.parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from detector import align_face
from src.config import Config
from src.hf.modeling_gend import GenD as GenD_HF
from src.model.GenD import GenD as GenD_Train
from src.retinaface import RetinaFace, prepare_model

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CKPT = "runs/rebuttal/wacv-DINOv3L-LN+L2+UA-U0.5-A0.5-seed0/checkpoints/best_mAP.ckpt"
HF_MODELS = [
    "yermandy/GenD_CLIP_L_14",
    "yermandy/GenD_PE_L",
    "yermandy/GenD_DINOv3_L",
]
OUTPUT_DIR = Path("outputs/tmp/gradio_app")


torch.set_float32_matmul_precision("high")


class DeepfakeDetector:
    """Handles model loading, caching, and inference for deepfake detection."""

    def __init__(self):
        self.model_cache: Dict[str, Dict] = {}
        self.detector_cache: Dict[float, RetinaFace] = {}

    def _get_dtype(self, precision: str) -> torch.dtype:
        """Determine torch dtype from precision string."""
        precision = (precision or "").lower()
        if DEVICE == "cpu":
            return torch.float32
        if "bf16" in precision:
            return torch.bfloat16
        if "16" in precision:
            return torch.float16
        return torch.float32

    def load_model(self, model_source: str, model_id: str) -> Tuple[Union[GenD_Train, GenD_HF], Callable, torch.dtype]:
        """Load and cache the GenD model."""
        cache_key = f"{model_source}::{model_id}::{DEVICE}"
        if cache_key in self.model_cache:
            return (
                self.model_cache[cache_key]["model"],
                self.model_cache[cache_key]["preproc"],
                self.model_cache[cache_key]["dtype"],
            )

        # Clear cache to free memory from previous models
        self.model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if model_source == "Hugging Face":
            model = GenD_HF.from_pretrained(model_id)
            model.eval()
            model.to(DEVICE)
            preproc = model.feature_extractor.preprocess
            dtype = torch.float32  # HF models usually float32 by default unless specified
        else:
            ckpt_path = model_id
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            ckpt = torch.load(ckpt_path, map_location="cpu")
            hparams = ckpt.get("hyper_parameters", {})
            precision = hparams.get("precision", "32-true")
            dtype = self._get_dtype(precision)

            config = Config(**hparams)
            model = GenD_Train(config)
            model.eval()
            model.load_state_dict(ckpt["state_dict"], strict=True)
            model.to(DEVICE)

            preproc = model.get_preprocessing()

        self.model_cache[cache_key] = {"model": model, "preproc": preproc, "dtype": dtype}
        return model, preproc, dtype

    def load_detector(self, face_thresh: float = 0.5) -> RetinaFace:
        """Load and cache the face detector."""
        face_thresh = float(face_thresh)
        if face_thresh in self.detector_cache:
            return self.detector_cache[face_thresh]
        model = prepare_model(face_thresh)
        self.detector_cache[face_thresh] = model
        return model

    def infer_faces(
        self,
        frame_bgr: np.ndarray,
        detector: RetinaFace,
        model: Union[GenD_Train, GenD_HF],
        preproc: Callable,
        dtype: torch.dtype,
        scale: float = 1.3,
        target_size: Optional[int] = None,
        max_faces: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, float]]:
        """Detect faces and run inference on them."""
        try:
            xyxy, landmarks = detector.detect(frame_bgr)
        except Exception:
            return []

        if xyxy is None or len(xyxy) == 0:
            return []

        # Select faces sorted by area (largest first) when limiting
        indices = list(range(len(xyxy)))
        indices.sort(key=lambda idx: (xyxy[idx][2] - xyxy[idx][0]) * (xyxy[idx][3] - xyxy[idx][1]), reverse=True)
        if max_faces is not None:
            indices = indices[: max(1, max_faces)]

        results = []
        for i in indices:
            lms = landmarks[i]
            try:
                aligned_face, _ = align_face(
                    frame_bgr,
                    lms,
                    target_size=(target_size, target_size) if target_size else None,
                    scale=scale,
                )
            except Exception:
                continue

            # Convert to PIL Image
            aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(aligned_face)

            with torch.no_grad():
                batch = preproc(pil_img).unsqueeze(0).to(DEVICE)
                if DEVICE == "cuda" and dtype in (torch.float16, torch.bfloat16):
                    batch = batch.to(dtype)

                out = model(batch)

                if isinstance(model, GenD_Train):
                    probs = out.logits_labels.softmax(dim=1).detach().cpu().numpy()[0]
                else:
                    # GenD_HF returns logits directly
                    probs = out.softmax(dim=-1).detach().cpu().numpy()[0]

                p_fake = float(probs[1])

            results.append((xyxy[i], p_fake))

        return results

    def annotate_frame(
        self, frame_bgr: np.ndarray, faces: List[Tuple[np.ndarray, float]], avg_fake: Optional[float] = None
    ) -> np.ndarray:
        """Annotate frame with bounding boxes and probabilities."""
        vis = frame_bgr.copy()
        for bbox, p_fake in faces:
            x1, y1, x2, y2 = map(int, bbox[:4])
            # Interpolate color from green (p_fake=0) to red (p_fake=1)
            blue = 0
            green = int(255 * (1 - p_fake))
            red = int(255 * p_fake)
            color = (blue, green, red)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            text = f"fake: {p_fake:.3f}"
            org = (x1 + 6, max(20, y1 + 20))
            cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        if avg_fake is not None:
            msg = f"Avg fake: {avg_fake:.3f}"
            org = (8, 28)
            cv2.putText(vis, msg, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, msg, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return vis


class MediaProcessor:
    """Handles processing of images and videos."""

    def __init__(self, detector: DeepfakeDetector):
        self.detector = detector

    def process_image(
        self,
        img_path: str,
        detector: RetinaFace,
        model: Union[GenD_Train, GenD_HF],
        preproc: Callable,
        dtype: torch.dtype,
        scale: float,
        target_size: Optional[int],
        out_dir: Path,
        max_faces: Optional[int] = None,
        progress_updater: Optional[Callable[[int], None]] = None,
    ) -> Tuple[str, Dict[str, float]]:
        """Process a single image."""
        try:
            img_rgb = iio.imread(img_path)
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {img_path} ({e})")

        faces = self.detector.infer_faces(img, detector, model, preproc, dtype, scale, target_size, max_faces)
        p_fake_vals = [pf for _, pf in faces]
        avg_fake = float(np.mean(p_fake_vals)) if p_fake_vals else 0.0
        med_fake = float(np.median(p_fake_vals)) if p_fake_vals else 0.0

        annotated = self.detector.annotate_frame(img, faces, avg_fake)
        out_path = out_dir / (Path(img_path).stem + "_annot.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)

        if progress_updater is not None:
            progress_updater(1)

        metrics = {
            "num_frames": 1,
            "num_faces": float(len(faces)),
            "avg_p_fake": avg_fake,
            "median_p_fake": med_fake,
        }
        return str(out_path), metrics

    def process_video(
        self,
        vid_path: str,
        detector: RetinaFace,
        model: Union[GenD_Train, GenD_HF],
        preproc: Callable,
        dtype: torch.dtype,
        scale: float,
        target_size: Optional[int],
        out_dir: Path,
        stride: int = 1,
        max_frames: int = -1,
        max_faces: Optional[int] = None,
        progress_updater: Optional[Callable[[int], None]] = None,
    ) -> Tuple[str, Dict[str, float]]:
        """Process a video."""
        try:
            meta = iio.immeta(vid_path, plugin="pyav")
            orig_fps = float(meta.get("fps", 25.0))
        except Exception:
            orig_fps = 25.0

        out_path = out_dir / (Path(vid_path).stem + "_annot.mp4")
        out_fps = max(1.0, orig_fps / max(1, stride))

        processed = 0
        frame_idx = 0
        p_fake_values: List[float] = []
        total_faces = 0
        writer = None

        try:
            for frame_rgb in iio.imiter(vid_path, plugin="pyav"):
                if frame_idx % stride != 0:
                    frame_idx += 1
                    continue

                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                if writer is None:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = imageio.get_writer(
                        str(out_path),
                        fps=out_fps,
                        codec="libx264",
                        quality=None,
                        pixelformat="yuv420p",
                        output_params=["-preset", "fast", "-crf", "23"],
                    )

                faces = self.detector.infer_faces(frame, detector, model, preproc, dtype, scale, target_size, max_faces)
                total_faces += len(faces)
                if faces:
                    p_fake_values.extend([pf for _, pf in faces])
                    running_avg = float(np.mean(p_fake_values))
                    vis = self.detector.annotate_frame(frame, faces, running_avg)
                else:
                    running_avg = float(np.mean(p_fake_values)) if p_fake_values else 0.0
                    vis = self.detector.annotate_frame(frame, [], running_avg)

                # Convert BGR to RGB for imageio
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                writer.append_data(vis_rgb)

                processed += 1
                frame_idx += 1
                if progress_updater is not None:
                    progress_updater(1)
                if max_frames != -1 and processed >= max_frames:
                    break
        finally:
            if writer is not None:
                writer.close()

        avg_fake = float(np.mean(p_fake_values)) if p_fake_values else 0.0
        med_fake = float(np.median(p_fake_values)) if p_fake_values else 0.0

        metrics = {
            "num_frames": float(processed),
            "num_faces": float(total_faces),
            "avg_p_fake": avg_fake,
            "median_p_fake": med_fake,
        }
        return str(out_path), metrics


def collect_inputs(files, folder_path: str) -> List[str]:
    """Collect valid media file paths from uploads and folder."""
    paths: List[str] = []
    if files:
        for f in files:
            p = getattr(f, "name", None) or getattr(f, "path", None) or str(f)
            if p and Path(p).suffix.lower() in VIDEO_EXTS.union(IMAGE_EXTS):
                paths.append(p)

    if folder_path:
        root = Path(folder_path)
        if root.is_dir():
            for ext in sorted(VIDEO_EXTS.union(IMAGE_EXTS)):
                paths.extend(str(p) for p in root.rglob(f"*{ext}"))

    # Deduplicate and sort
    seen = set()
    dedup = []
    for p in paths:
        if p not in seen:
            dedup.append(p)
            seen.add(p)
    return dedup


def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


DETECTOR = DeepfakeDetector()


def run_inference(
    model_source: str,
    hf_model: str,
    local_ckpt: str,
    files,
    # folder_path: str,
    face_thresh: float,
    stride: int,
    max_frames: int,
    scale: float,
    target_size: Optional[int],
    max_faces: int,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    """Main inference function for Gradio."""
    if target_size == -1:
        target_size = None

    detector_obj = DETECTOR
    processor = MediaProcessor(detector_obj)

    print("Loading model...")
    yield (
        pd.DataFrame(columns=["input", "num_frames", "num_faces", "avg_p_fake", "median_p_fake"]),
        "### ⏳ Status: Loading model...",
        None,
        None,
    )

    model_id = hf_model if model_source == "Hugging Face" else local_ckpt
    model, preproc, dtype = detector_obj.load_model(model_source, model_id)

    print("Loading face detector...")
    yield (
        pd.DataFrame(columns=["input", "num_frames", "num_faces", "avg_p_fake", "median_p_fake"]),
        "### ⏳ Status: Loading face detector...",
        None,
        None,
    )
    detector = detector_obj.load_detector(face_thresh)

    print("Collecting inputs...")
    yield (
        pd.DataFrame(columns=["input", "num_frames", "num_faces", "avg_p_fake", "median_p_fake"]),
        "### ⏳ Status: Collecting inputs...",
        None,
        None,
    )

    inputs = collect_inputs(files, None)
    if not inputs:
        empty_df = pd.DataFrame(columns=["input", "num_frames", "num_faces", "avg_p_fake", "median_p_fake"])
        yield (
            empty_df,
            "### ❌ Status: No valid inputs found.",
            None,
            None,
        )
        return

    # Calculate total progress units (frames for videos, 1 for images)
    print("Calculating total progress...")
    yield (
        pd.DataFrame(columns=["input", "num_frames", "num_faces", "avg_p_fake", "median_p_fake"]),
        "### ⏳ Status: Calculating total progress...",
        None,
        None,
    )
    total_progress_units = 0
    for p in inputs:
        if is_image(p):
            total_progress_units += 1
        elif is_video(p):
            try:
                props = iio.improps(p, plugin="pyav")
                frame_count = props.shape[0]
                processed_frames = frame_count // max(1, stride)
                if max_frames != -1:
                    processed_frames = min(max_frames, processed_frames)
                total_progress_units += max(1, processed_frames)
            except Exception:
                total_progress_units += 1

    total_progress_units = max(1, total_progress_units)

    current_progress = 0

    def advance_progress(step: int = 1) -> None:
        nonlocal current_progress
        current_progress = min(total_progress_units, current_progress + step)
        fraction = current_progress / total_progress_units if total_progress_units else 1.0
        progress(
            fraction,
            desc=f"Processing frames ({current_progress}/{total_progress_units})",
        )

    progress(0.0, desc=f"Processing frames (0/{total_progress_units})")
    print("Starting inference...")
    yield (
        pd.DataFrame(columns=["input", "num_frames", "num_faces", "avg_p_fake", "median_p_fake"]),
        "### 🚀 Status: Starting inference...",
        None,
        None,
    )

    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup directories
    inputs_dir = OUTPUT_DIR / "inputs"
    outputs_dir = OUTPUT_DIR / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    output_files = []
    processed_inputs = []

    for idx, p in enumerate(inputs):
        # Copy input to inputs_dir
        try:
            p_path = Path(p)
            unique_name = f"{p_path.stem}_{uuid.uuid4().hex[:8]}{p_path.suffix}"
            new_input_path = inputs_dir / unique_name
            shutil.copy2(p, new_input_path)
            p = str(new_input_path)
        except Exception as e:
            print(f"Failed to copy input {p}: {e}")

        processed_inputs.append(p)

        try:
            if is_video(p):
                out_p, metrics = processor.process_video(
                    p,
                    detector,
                    model,
                    preproc,
                    dtype,
                    scale,
                    target_size,
                    outputs_dir,
                    stride,
                    max_frames,
                    max_faces if max_faces > 0 else None,
                    advance_progress,
                )
            elif is_image(p):
                out_p, metrics = processor.process_image(
                    p,
                    detector,
                    model,
                    preproc,
                    dtype,
                    scale,
                    target_size,
                    outputs_dir,
                    max_faces if max_faces > 0 else None,
                    advance_progress,
                )
            else:
                continue

            rows.append({"input": p, "output": out_p, **metrics})
            output_files.append(out_p)

        except Exception as e:
            print(f"Error processing {p}: {e}")
            rows.append(
                {
                    "input": p,
                    "output": "",
                    "num_frames": 0,
                    "num_faces": 0,
                    "avg_p_fake": 0.0,
                    "median_p_fake": 0.0,
                    "error": str(e),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty and "input" in df.columns:
        df = df.sort_values("input").reset_index(drop=True)

    # Log to CSV
    log_file = OUTPUT_DIR / "inference_log.csv"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_file.exists()
    df.to_csv(log_file, mode="a", header=write_header, index=False)

    # Prepare display DataFrame
    display_df = df.copy()
    if not display_df.empty:
        display_df["input"] = display_df["input"].apply(lambda x: Path(x).name)
        if "output" in display_df.columns:
            display_df = display_df.drop(columns=["output"])

    final_status = "### ✅ Status: Inference complete!\n\n"

    # summary = []
    # if not df.empty and "avg_p_fake" in df.columns:
    #     overall_avg = float(df["avg_p_fake"].mean())
    #     overall_med = float(df["median_p_fake"].median())
    # summary.append(f"**Overall avg fake:** {overall_avg:.4f}")
    # summary.append(f"**Overall median fake:** {overall_med:.4f}")
    # final_status += "\n\n".join(summary)

    progress(1.0, desc=f"Processing frames ({total_progress_units}/{total_progress_units})")

    print("Inference complete!")
    yield (
        display_df,
        final_status,
        processed_inputs,
        output_files,
    )


def get_thumbnail(path: str) -> Optional[str]:
    """Get thumbnail image path for preview (image itself or first frame of video)."""
    if is_image(path):
        return path
    if is_video(path):
        return path
    return None


def get_all_inputs(files, folder_path):
    """Get all input paths for preview."""
    return collect_inputs(files, folder_path)


def build_ui():
    """Build the Gradio interface."""
    with gr.Blocks(title="Deepfake Detector", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🕵️‍♂️ Deepfake Detector
            Upload images/videos or specify a folder to process all media files.
            Detects faces, runs deepfake analysis, and visualizes results.
            """
        )

        with gr.Group():
            with gr.Row():
                with gr.Column(scale=1, min_width=352):
                    model_source = gr.Radio(
                        ["Hugging Face", "Local Checkpoint"], label="Model Source", value="Hugging Face"
                    )
                with gr.Column(scale=2):
                    hf_model = gr.Dropdown(HF_MODELS, label="HF Model", value=HF_MODELS[-1], visible=True)
                    local_ckpt = gr.Textbox(label="Checkpoint path", value=DEFAULT_CKPT, visible=False)

        with gr.Row():
            files = gr.Files(label="Upload files", file_count="multiple")
            # folder = gr.Textbox(label="Folder path (optional)", placeholder="/path/to/images_or_videos")

        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                face_thresh = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Face detection threshold")
                scale = gr.Slider(1.0, 2.0, value=1.3, step=0.05, label="Face align scale")
                target_size = gr.Number(value=-1, precision=0, label="Face size (px) (-1=original)")

            with gr.Row():
                stride = gr.Slider(1, 10, value=1, step=1, label="Frame stride (video)")
                max_frames = gr.Number(value=-1, precision=0, label="Max frames per video (-1=all)")
                max_faces = gr.Slider(1, 10, value=1, step=1, label="Max faces per frame")

        run_btn = gr.Button("🚀 Run Detection", variant="primary", size="lg")

        status_summary = gr.Markdown()

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📷 Input Preview")
                input_gallery = gr.Gallery(
                    label="Input Preview",
                    show_label=False,
                    columns=1,
                    object_fit="contain",
                    height="auto",
                    preview=True,
                    selected_index=0,
                )
            with gr.Column():
                gr.Markdown("### 🎯 Output Preview")
                output_gallery = gr.Gallery(
                    label="Output Preview",
                    show_label=False,
                    columns=1,
                    object_fit="contain",
                    height="auto",
                    preview=True,
                    selected_index=0,
                )

        with gr.Row():
            gr.Markdown("### 📊 Results")
            copy_btn = gr.Button("📋 Copy to Clipboard", size="sm", scale=0)
            export_btn = gr.Button("💾 Export to CSV", size="sm", scale=0)

        table = gr.Dataframe(
            headers=["input", "num_frames", "num_faces", "avg_p_fake", "median_p_fake"],
            wrap=True,
            interactive=False,
        )

        copy_btn.click(
            fn=None,
            inputs=[table],
            js="""(table_data) => {
                if (!table_data) return;
                const headers = table_data.headers;
                const data = table_data.data;
                if (!headers || !data) return;
                let text = headers.join(",") + "\\n";
                data.forEach(row => {
                    text += row.join(",") + "\\n";
                });
                navigator.clipboard.writeText(text);
            }""",
        )

        export_btn.click(
            fn=None,
            inputs=[table],
            js="""(table_data) => {
                if (!table_data) return;
                const headers = table_data.headers;
                const data = table_data.data;
                if (!headers || !data) return;
                let text = headers.join(",") + "\\n";
                data.forEach(row => {
                    text += row.join(",") + "\\n";
                });
                const blob = new Blob([text], { type: 'text/csv' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'results.csv';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }""",
        )

        def update_model_input(source):
            if source == "Hugging Face":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        model_source.change(fn=update_model_input, inputs=model_source, outputs=[hf_model, local_ckpt])

        run_btn.click(
            fn=run_inference,
            inputs=[
                model_source,
                hf_model,
                local_ckpt,
                files,
                # folder,
                face_thresh,
                stride,
                max_frames,
                scale,
                target_size,
                max_faces,
            ],
            outputs=[
                table,
                status_summary,
                input_gallery,
                output_gallery,
            ],
        )

        # Update input preview on change
        def update_previews(files_in, folder_in=None):
            return get_all_inputs(files_in, folder_in)

        files.change(
            fn=update_previews,
            inputs=[
                files,
                # folder,
            ],
            outputs=input_gallery,
        )
        # folder.change(fn=update_previews, inputs=[files, folder], outputs=input_gallery)

    return demo


if __name__ == "__main__":
    ui = build_ui()
    returns = ui.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        # share=True,
    )
    print("Gradio UI launched. Returns:", returns)
