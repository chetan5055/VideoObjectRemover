import tempfile
from pathlib import Path
import traceback
import subprocess
import hashlib
import os

import streamlit as st

from sorawm.core import SoraWM
from sorawm.schemas import CleanerType


def _pick_device() -> str:
    """Pick best device without crashing if CUDA isn't available."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _make_browser_preview_mp4(input_path: Path, out_path: Path):
    """
    Make a browser-safe MP4 preview (H.264 + AAC).
    Fast encode intended ONLY for preview.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", "scale='min(1280,iw)':-2",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    st.set_page_config(page_title="Sora Watermark Cleaner", page_icon="🎬", layout="centered")

    st.markdown(
        """
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='margin-bottom: 0.5rem;'>🎬 Sora Watermark Cleaner by Chetan Narayana</h1>
            <p style='font-size: 1.2rem; color: #666; margin-bottom: 1rem;'>
                Remove watermarks from Sora-generated videos with AI-powered precision
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style='text-align: center; padding: 1rem 0; margin-top: 0.5rem;'>
            <p style='color: #888; font-size: 0.9rem;'>
                Built with ❤️ by Chetan using Streamlit and AI
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ---------------------------
    # Model selection section
    # ---------------------------
    st.markdown("### ⚙️ Model Settings")

    col1, col2 = st.columns([2, 3])

    MODEL_LABELS = {
        CleanerType.LAMA: "🚀 LAMA (Fast, Good Quality)",
        CleanerType.E2FGVI_HQ: "💎 E2FGVI-HQ (Best Quality, Temporal Consistency)",
    }
    if hasattr(CleanerType, "PROPAINTER"):
        MODEL_LABELS[CleanerType.PROPAINTER] = "🧠 ProPainter (Best Quality, Slowest)"

    MODEL_INFO = {
        CleanerType.LAMA: (
            "⚡ **Fast processing** — Recommended for most videos.\n\n"
            "Uses **LaMa (Large Mask Inpainting)** for quick watermark removal."
        ),
        CleanerType.E2FGVI_HQ: (
            "🎯 **Highest quality** — Temporal flow-based video inpainting.\n\n"
            "Best for professional results. **Slow on CPU**, good on GPU.\n\n"
            "**Time consistency is guaranteed.**"
        ),
    }
    if hasattr(CleanerType, "PROPAINTER"):
        MODEL_INFO[CleanerType.PROPAINTER] = (
            "🧩 **Best quality (when masks are good)** — Strong temporal consistency.\n\n"
            "Slower, but usually cleaner output on complex motion.\n\n"
            "⚠️ Requires ProPainter repo path + (optional) weights dir.\n\n"
            "✅ Use **Fast Mode** on 8GB VRAM to avoid CUDA OOM."
        )

    with col1:
        options = [CleanerType.LAMA, CleanerType.E2FGVI_HQ]
        if hasattr(CleanerType, "PROPAINTER"):
            options.append(CleanerType.PROPAINTER)

        model_type = st.selectbox(
            "Select Cleaner Model:",
            options=options,
            format_func=lambda x: MODEL_LABELS.get(x, str(x)),
        )

    with col2:
        st.info(MODEL_INFO.get(model_type, "Model information not available."))

    # ---------------------------
    # ProPainter extra inputs
    # ---------------------------
    propainter_dir = None
    propainter_weights_dir = None
    propainter_fast_mode = True

    if hasattr(CleanerType, "PROPAINTER") and model_type == CleanerType.PROPAINTER:
        st.markdown("#### 🧠 ProPainter Setup")

        # IMPORTANT: default to RunPod/Linux path, not Windows
        propainter_dir = st.text_input(
            "ProPainter Repo Path",
            value="/workspace/ProPainter",
            help="RunPod/Linux example: /workspace/ProPainter (not a Windows path).",
        )

        propainter_weights_dir = st.text_input(
            "Weights Dir (optional)",
            value="",
            help="Optional. If empty, ProPainter usually uses its own ./weights folder.",
        )

        propainter_fast_mode = st.checkbox(
            "⚡ Fast Mode (Recommended for 8GB VRAM)",
            value=True,
            help="Reduces VRAM usage + speeds up processing. Helps prevent CUDA OOM on 8GB GPUs.",
        )

        if propainter_weights_dir is not None and propainter_weights_dir.strip() == "":
            propainter_weights_dir = None

    # ---------------------------
    # Lazy init engine (NEVER blocks UI)
    # ---------------------------
    device = _pick_device()
    st.caption(f"🖥️ Device: **{device.upper()}**")

    def build_sora_kwargs():
        sora_kwargs = {"cleaner_type": model_type, "device": device}

        if hasattr(CleanerType, "PROPAINTER") and model_type == CleanerType.PROPAINTER:
            sora_kwargs.update(
                {
                    "propainter_dir": propainter_dir,
                    "propainter_weights_dir": propainter_weights_dir,
                    "propainter_fast_mode": propainter_fast_mode,
                }
            )
        return sora_kwargs

    def get_engine_safe(sora_kwargs):
        try:
            return SoraWM(**sora_kwargs), None
        except FileNotFoundError as e:
            return None, str(e)
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    def ensure_engine():
        sora_kwargs = build_sora_kwargs()

        # Block Windows path in RunPod/Linux
        if hasattr(CleanerType, "PROPAINTER") and model_type == CleanerType.PROPAINTER:
            prop_dir = (propainter_dir or "").strip()
            if ":" in prop_dir and "\\" in prop_dir:
                return None, "❌ You pasted a Windows path. On RunPod/Linux use: /workspace/ProPainter"

            # quick helpful checks
            if prop_dir and not Path(prop_dir).exists():
                return None, f"❌ ProPainter repo path not found: {prop_dir}\n\nDid you clone it in RunPod? Try: `git clone https://github.com/sczhou/ProPainter.git /workspace/ProPainter`"

            if prop_dir and not Path(prop_dir, "inference_propainter.py").exists():
                return None, f"❌ inference_propainter.py not found inside: {prop_dir}\n\nMake sure `propainter_dir` points to the ProPainter repo ROOT."

        settings_key = (
            model_type,
            propainter_dir,
            propainter_weights_dir,
            propainter_fast_mode,
            device,
        )

        if (
            "sora_wm" in st.session_state
            and st.session_state.get("settings_key") == settings_key
            and st.session_state.get("sora_wm") is not None
        ):
            return st.session_state.sora_wm, None

        with st.spinner(f"Loading {model_type.value.upper()} model..."):
            engine, err = get_engine_safe(sora_kwargs)

        st.session_state.settings_key = settings_key
        st.session_state.sora_wm = engine

        return engine, err

    # Optional manual load button (nice for testing)
    colA, colB = st.columns([1, 2])
    with colA:
        if st.button("🔄 Load / Reload Model", use_container_width=True):
            engine, err = ensure_engine()
            if err:
                st.error(err)
            else:
                st.success(f"✅ {model_type.value.upper()} model loaded!")
    with colB:
        if hasattr(CleanerType, "PROPAINTER") and model_type == CleanerType.PROPAINTER:
            st.caption("RunPod path should be like **/workspace/ProPainter**")

    st.markdown("---")

    mode = st.radio("Select input mode:", ["📁 Upload Video File", "🗂️ Process Folder"], horizontal=True)

    # ===========================
    # SINGLE VIDEO MODE
    # ===========================
    if mode == "📁 Upload Video File":
        uploaded_file = st.file_uploader(
            "Upload your video",
            type=["mp4", "avi", "mov", "mkv"],
            accept_multiple_files=False,
            key="single_uploader",
        )

        if uploaded_file:
            video_bytes = uploaded_file.getvalue()

            # reset if new file
            if st.session_state.get("current_file_name") != uploaded_file.name:
                st.session_state.current_file_name = uploaded_file.name
                st.session_state.input_video_bytes = video_bytes
                st.session_state.pop("processed_video_data", None)
                st.session_state.pop("processed_video_name", None)
                st.session_state.pop("preview_video_bytes", None)
                st.session_state.pop("preview_sig", None)

            st.success(f"✅ Uploaded: {uploaded_file.name}")

            # stable signature so we don't re-encode preview every rerun
            sig = hashlib.md5(st.session_state.input_video_bytes).hexdigest()

            if st.session_state.get("preview_sig") != sig:
                with st.spinner("Preparing preview (browser-compatible)..."):
                    with tempfile.TemporaryDirectory() as td:
                        tmp = Path(td)
                        src = tmp / uploaded_file.name
                        src.write_bytes(st.session_state.input_video_bytes)

                        preview_path = tmp / "preview.mp4"

                        try:
                            _make_browser_preview_mp4(src, preview_path)
                            st.session_state.preview_video_bytes = preview_path.read_bytes()
                            st.session_state.preview_sig = sig
                        except Exception:
                            st.session_state.preview_video_bytes = st.session_state.input_video_bytes
                            st.session_state.preview_sig = sig

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("### 📥 Original Video")
                st.video(st.session_state.preview_video_bytes, format="video/mp4")

            with col_right:
                st.markdown("### 🎬 Processed Video")
                if "processed_video_data" not in st.session_state:
                    st.info("Click 'Remove Watermark' to process the video")
                else:
                    st.video(st.session_state.processed_video_data, format="video/mp4")

            if st.button("🚀 Remove Watermark", type="primary", use_container_width=True):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)

                    try:
                        # ✅ lazy load engine here (never blocks uploader)
                        engine, err = ensure_engine()
                        if err:
                            st.error(err)
                            st.stop()

                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def update_progress(progress: int):
                            progress = max(0, min(100, int(progress)))
                            progress_bar.progress(progress / 100)
                            if progress < 50:
                                status_text.text(f"🔍 Detecting watermarks... {progress}%")
                            elif progress < 95:
                                status_text.text(f"🧹 Removing watermarks... {progress}%")
                            else:
                                status_text.text(f"🎵 Merging audio... {progress}%")

                        input_path = tmp_path / uploaded_file.name
                        input_path.write_bytes(st.session_state.input_video_bytes)

                        output_path = tmp_path / f"cleaned_{uploaded_file.name}"

                        engine.run(input_path, output_path, progress_callback=update_progress)

                        st.session_state.processed_video_data = output_path.read_bytes()
                        st.session_state.processed_video_name = f"cleaned_{uploaded_file.name}"

                        st.success("✅ Watermark removed successfully!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error processing video: {str(e)}")
                        st.code(traceback.format_exc())

                        # Helpful hint for broken pipe
                        if "Broken pipe" in str(e) or "Errno 32" in str(e):
                            st.warning(
                                "Broken pipe usually means FFmpeg output encoder crashed.\n"
                                "You MUST patch `sorawm/core.py` to capture ffmpeg stderr.\n"
                                "I gave you the exact patch below."
                            )

            if "processed_video_data" in st.session_state:
                st.download_button(
                    label="⬇️ Download Cleaned Video",
                    data=st.session_state.processed_video_data,
                    file_name=st.session_state.processed_video_name,
                    mime="video/mp4",
                    use_container_width=True,
                )

    # ===========================
    # FOLDER / MULTI VIDEO MODE
    # ===========================
    else:
        st.info("💡 Upload multiple videos and process them in one go.")

        uploaded_files = st.file_uploader(
            "Upload videos",
            type=["mp4", "avi", "mov", "mkv"],
            accept_multiple_files=True,
            key="folder_uploader",
        )

        if uploaded_files:
            video_count = len(uploaded_files)
            st.success(f"✅ {video_count} video file(s) uploaded")

            if st.button("🚀 Process All Videos", type="primary", use_container_width=True):
                with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    input_folder = tmp_path / "input"
                    output_folder = tmp_path / "output"
                    input_folder.mkdir(exist_ok=True)
                    output_folder.mkdir(exist_ok=True)

                    try:
                        # ✅ lazy load engine here too
                        engine, err = ensure_engine()
                        if err:
                            st.error(err)
                            st.stop()

                        for up in uploaded_files:
                            (input_folder / up.name).write_bytes(up.getvalue())

                        st.session_state.batch_processed_files = []
                        progress_bar = st.progress(0)
                        current_file_text = st.empty()

                        processed_count = 0

                        def update_progress(progress: int):
                            p = max(0, min(100, int(progress)))
                            overall = (processed_count + (p / 100.0)) / float(video_count)
                            progress_bar.progress(min(1.0, max(0.0, overall)))

                            if p < 50:
                                current_file_text.text(
                                    f"🔍 File {processed_count + 1}/{video_count}: Detecting... {p}%"
                                )
                            elif p < 95:
                                current_file_text.text(
                                    f"🧹 File {processed_count + 1}/{video_count}: Cleaning... {p}%"
                                )
                            else:
                                current_file_text.text(
                                    f"🎵 File {processed_count + 1}/{video_count}: Merging audio... {p}%"
                                )

                        for video_file in sorted(input_folder.glob("*")):
                            if not video_file.is_file():
                                continue
                            if video_file.suffix.lower() not in [".mp4", ".avi", ".mov", ".mkv"]:
                                continue

                            output_path = output_folder / f"cleaned_{video_file.name}"
                            engine.run(video_file, output_path, progress_callback=update_progress)

                            processed_count += 1
                            st.session_state.batch_processed_files.append(
                                {"name": output_path.name, "data": output_path.read_bytes()}
                            )

                        progress_bar.progress(1.0)
                        current_file_text.text("✅ All videos processed!")
                        st.success("✅ All videos processed successfully!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error processing videos: {str(e)}")
                        st.code(traceback.format_exc())

                        if "Broken pipe" in str(e) or "Errno 32" in str(e):
                            st.warning(
                                "Broken pipe usually means FFmpeg output encoder crashed.\n"
                                "Patch `sorawm/core.py` to capture ffmpeg stderr (patch below)."
                            )

            if "batch_processed_files" in st.session_state and st.session_state.batch_processed_files:
                st.markdown("---")
                st.markdown("### ⬇️ Download Processed Videos")

                for i, item in enumerate(st.session_state.batch_processed_files):
                    cols = st.columns([3, 1])
                    with cols[0]:
                        st.write(f"📹 {item['name']}")
                    with cols[1]:
                        st.download_button(
                            label="⬇️ Download",
                            data=item["data"],
                            file_name=item["name"],
                            mime="video/mp4",
                            key=f"dl_{i}_{item['name']}",
                            use_container_width=True,
                        )


if __name__ == "__main__":
    main()
