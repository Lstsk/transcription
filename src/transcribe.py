"""
Video Transcription with Speaker Diarization

Extracts audio from video files and produces speaker-labeled transcripts
using WhisperX (Whisper + PyAnnote). All processing runs locally.
"""

from typing import List, Dict
import os
import gc
from pathlib import Path
import subprocess
import warnings
import math
import wave
import contextlib

import torch
import whisperx
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline
from datetime import timedelta, datetime
import logging
import sys
import time
import argparse
import csv

# ---------------------------------------------------------------------------
# Constants 
# ---------------------------------------------------------------------------
DATA_ROOT = "data"


# ---------------------------------------------------------------------------
# PyTorch 2.6+ changed torch.load to default to weights_only=True, which
# breaks loading older PyAnnote model checkpoints. We patch torch.load to
# restore the previous default until PyAnnote ships compatible weights.
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", message=".*weights_only.*")
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load



# ---------------------------------------------------------------------------
# Utility functions for batch processing
# ---------------------------------------------------------------------------
def find_webm_files(base_dir: str) -> List[Path]:    
    base = Path(base_dir)
    return list(base.rglob("*.webm"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_audio_to_temp(webm_path: Path, temp_root: str = "data/temp", out_format: str = "wav") -> Path:
    webm_path = Path(webm_path)
    temp_dir = Path(temp_root)
    ensure_dir(temp_dir)
    out_name = webm_path.stem + "." + out_format
    out_path = temp_dir / out_name

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(webm_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return out_path

def extract_audio(video_path: str) -> str:
    """Extract audio from a video file using ffmpeg and save as WAV."""
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    command = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-af", "highpass=f=80, afftdn=nr=12:nf=-50:tn=1, loudnorm",
        audio_path, "-y"
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return audio_path



# ---------------------------------------------------------------------------
# Main transcription and diarization logic
# ---------------------------------------------------------------------------
def transcribe_and_diarize(audio_path: str, hf_token: str, model, align_model, metadata, diarize_model,
                           no_diarize: bool = False, no_align: bool = False) -> list[dict]:
    """
    Run the full WhisperX pipeline:
      1. Transcribe with Whisper
      2. Align word-level timestamps
      3. Diarize speakers with PyAnnote
      4. Assign speaker labels to segments
    """
    device = "cpu"  # Safest default for Mac; change to "cuda" on GPU machines
    
    logger = logging.getLogger("transcribe")
    # --- Step 1: Transcribe ---
    logger.info("1. Transcribing audio...")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)

    segments = result.get("segments", [])

    # --- Step 2: Align timestamps ---
    if not no_align:
        logger.info("2. Aligning timestamps...")
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, device,
            return_char_alignments=False,
        )
        segments = result.get("segments", [])
    else:
        logger.info("2. Skipping alignment (flag)")

    # --- Step 3: Speaker diarization ---
    if not no_diarize:
        logger.info("3. Performing speaker diarization...")
        diarize_segments = diarize_model(audio, min_speakers = 2, max_speakers = 2)
        # --- Step 4: Assign speakers to segments ---
        result = whisperx.assign_word_speakers(diarize_segments, result)
        segments = result.get("segments", [])
    else:
        logger.info("3. Skipping diarization (flag)")

    return segments


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    try:
        seconds = float(seconds or 0.0)
    except Exception:
        seconds = 0.0
    total_seconds = int(seconds)
    ms = int((seconds - total_seconds) * 1000)
    td = timedelta(seconds=total_seconds)
    return f"{str(td)}.{ms:03d}"


def print_output(segment):
    logger = logging.getLogger("transcribe")
    for seg in segments:  # Note: segments variable needs to be accessible here if used
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        start = seg.get("start", seg.get("start_time", 0.0))
        ts = format_timestamp(start)
        logger.info(f"[{ts}] [{speaker}]: {text}")

def save_into_csv(segments, output_file):
    logger = logging.getLogger("transcribe")
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['start_time', 'end_time', 'speaker', 'text', 'confidence']
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)

        writer.writeheader()

        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            text = seg.get("text", "").strip()
            start = seg.get("start", seg.get("start_time", 0.0))
            end = seg.get("end", seg.get("end_time", 0.0))
            stf = format_timestamp(start)
            endf = format_timestamp(end)
            logger.info(f"[{stf}] [{endf}] [{speaker}]: {text}")
            avg_prob = seg.get("avg_logprob", None)
            if avg_prob is not None: 
                confidence = round(math.exp(avg_prob), 3)
            else: 
                confidence = ""
            writer.writerow({'start_time': stf, 'end_time': endf,"speaker": speaker, "text": text, "confidence": confidence})



def append_processing_log(output_dir: Path, file_name: str, transcribed: bool, diarized: bool, transcribe_time_sec: float):
    """Append a summary line to processing_log.csv in the given output dir."""
    try:
        ensure_dir(output_dir)
        log_path = output_dir / "processing_log.csv"
        header = ['file_name', 'transcribed', 'diarized', 'transcribe_time_sec', 'timestamp']
        write_header = not log_path.exists()
        minutes, seconds = divmod(transcribe_time_sec, 60)
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([file_name, str(transcribed), str(diarized),  f"{minutes:.0f}:{seconds:.0f}", ts])
    except Exception as e:
        logging.getLogger('transcribe').warning(f"Failed to write processing log: {e}")


def get_output_path(webm_path: Path, output_dir: str = "output") -> Path:
    """Generates the expected CSV path for a given webm file."""
    # This keeps the filename but puts it in the output folder
    file_name = webm_path.stem + "_transcript.csv"
    return Path(output_dir) / file_name


def main():
    load_dotenv(override=True)
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN not set.")
        print("Create a .env file with: HF_TOKEN=your_token_here")
        return
    
    # 1. Parse arguments FIRST
    parser = argparse.ArgumentParser(description="Transcribe .webm files with optional diarization/align")
    parser.add_argument("--no-diarize", action="store_true", help="Skip speaker diarization")
    parser.add_argument("--no-align", action="store_true", help="Skip timestamp alignment")
    parser.add_argument("--input", "-i", help="Path to a single .webm file to process (overrides data root)")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory for CSVs and logs")
    parser.add_argument("--timestamped-log", action="store_true", help="Create a timestamped log file per run")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (DEBUG)")
    args = parser.parse_args()

    # Ensure output directory exists immediately after parsing args
    output_base = Path(args.output_dir)
    ensure_dir(output_base)

    # 2. Setup devices
    compute_type = "int8"
    batch_size = 16
    device = 'cpu'
    asr_options = {"beam_size" : 1, "patience": 1}
    
    # 3. Load base models
    model = whisperx.load_model("large-v2", device, compute_type=compute_type, asr_options = asr_options)
    
    if not args.no_align:
        align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
    else:
        align_model, metadata = None, None

    # 4. Conditionally load PyAnnote ONLY if we need it
    if not args.no_diarize:
        diarize_model = DiarizationPipeline(device=device)
    else:
        diarize_model = None

    # Configure logging: suppress noisy libraries and log to console + file
    logging.getLogger().setLevel(logging.WARNING)
    for lib in ("transformers", "whisperx", "torch", "torchaudio", "pyannote", "ffmpeg"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    logger = logging.getLogger("transcribe")
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler: fixed name or timestamped per run
    if args.timestamped_log:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(str(output_base / f"transcription_{run_ts}.log"), mode="a")
    else:
        fh = logging.FileHandler(str(output_base / "transcription.log"), mode="a")
    fh.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
 
    # Determine files to process: single input or all found
    if args.input:
        all_files = [Path(args.input)]
    else:
        all_files = find_webm_files(DATA_ROOT)
    total_files = len(all_files)
    
    logger.info(f"Found {total_files} files to process.")

    # Process all .webm files in the data directory
    for idx, webm in enumerate(all_files, 1):
        # Pass the output directory to get_output_path
        output_csv = get_output_path(webm, args.output_dir)

        # Resume capability: skip if output already exists
        if output_csv.exists():
            logger.info(f"[{idx}/{total_files}] Skipping: {webm.name} (Output already exists)")
            continue

        # Process the file
        try:
            logger.info(f"\n[{idx}/{total_files}] Processing: {webm.name}")
            start_time = time.time()
            audio_path = extract_audio_to_temp(webm)
            try:
                result = transcribe_and_diarize(
                    audio_path=audio_path, 
                    hf_token=hf_token, 
                    model=model, 
                    align_model=align_model, 
                    metadata=metadata, 
                    diarize_model=diarize_model,
                    no_diarize=args.no_diarize,  
                    no_align=args.no_align      
                )
                save_into_csv(result, str(output_csv))
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            
            
            elapsed = time.time() - start_time
            # Append a short per-file processing summary to processing_log.csv
            try:
                append_processing_log(output_base, webm.name, transcribed=True, diarized=(not args.no_diarize), transcribe_time_sec=elapsed)

            except Exception:
                logger.warning(f"Failed to write processing logs for {webm.name}")

            logger.info(f"[{idx}/{total_files}] Processed {webm.name} in {elapsed:.2f}s; saved to {output_csv}")
        except Exception as e:
            logger.exception(f"processing failed for {webm}: {e}")


if __name__ == "__main__":
    main()