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

import torch
import whisperx
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline
from datetime import timedelta
from datetime import timedelta

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
    subprocess.run(cmd, check=True)
    return out_path

def extract_audio(video_path: str) -> str:
    """Extract audio from a video file using ffmpeg and save as WAV."""
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    command = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-af", "highpass=f=80, afftdn=nr=12:nf=-50:tn=1, loudnorm",
        # "-ar", "44100", "-ac", "2",
        audio_path, "-y"
    ]
    subprocess.run(command, check=True)
    return audio_path



# ---------------------------------------------------------------------------
# Main transcription and diarization logic
# ---------------------------------------------------------------------------
def transcribe_and_diarize(audio_path: str, hf_token: str, model, align_model, metadata, diarize_model) -> list[dict]:
    """
    Run the full WhisperX pipeline:
      1. Transcribe with Whisper
      2. Align word-level timestamps
      3. Diarize speakers with PyAnnote
      4. Assign speaker labels to segments
    """
    device = "cpu"  # Safest default for Mac; change to "cuda" on GPU machines
    
    # --- Step 1: Transcribe ---
    print("1. Transcribing audio...")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)


    # --- Step 2: Align timestamps ---
    print("2. Aligning timestamps...")
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False,
    )


    # --- Step 3: Speaker diarization ---
    print("3. Performing speaker diarization...")

    diarize_segments = diarize_model(audio, min_speakers = 2, max_speakers = 2)

    # --- Step 4: Assign speakers to segments ---
    result = whisperx.assign_word_speakers(diarize_segments, result)

    return result["segments"]


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
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        start = seg.get("start", seg.get("start_time", 0.0))
        ts = format_timestamp(start)
        print(f"[{ts}] [{speaker}]: {text}")

def save_into_csv(segments, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['start_time', 'end_time', 'speaker', 'text']
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)

        writer.writeheader()

        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            text = seg.get("text", "").strip()
            start = seg.get("start", seg.get("start_time", 0.0))
            end = seg.get("end", seg.get("end_time", 0.0))
            stf = format_timestamp(start)
            endf = format_timestamp(end)
            print(f"[{stf}] [{endf}] [{speaker}]: {text}")
            writer.writerow({'start_time': stf, 'end_time': endf,"speaker": speaker, "text": text})

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
    
    compute_type = "int8"
    batch_size = 16
    device = 'cpu'
    asr_options = {"beam_size" : 1, "patience": 1}
    model = whisperx.load_model("turbo", device, compute_type=compute_type, asr_options = asr_options)
    align_model, metadata = whisperx.load_align_model(language_code="en", device=device)

    diarize_model = DiarizationPipeline(
        use_auth_token=hf_token, device=device
    )
    # Ensure output directory exists
    output_base = Path("output")
    ensure_dir(output_base)
 
    all_files = find_webm_files(DATA_ROOT)
    total_files = len(all_files)
    
    print(f"Found {total_files} files to process.")

    # Process all .webm files in the data directory
    for idx, webm in enumerate(all_files, 1):
        output_csv = get_output_path(webm)

        # Resume capability: skip if output already exists
        if output_csv.exists():
            print(f"[{idx}/{total_files}] Skipping: {webm.name} (Output already exists)")
            continue

        # Process the file
        try:
            print(f"\n[{idx}/{total_files}] Processing: {webm.name}")
            audio_path = extract_audio_to_temp(webm)
            result = transcribe_and_diarize(audio_path, hf_token, model, align_model, metadata, diarize_model)

            save_into_csv(result, str(output_csv))
            print(f"Successfully saved to {output_csv}")
        except Exception as e:
            print(f"processing failed for {webm}: {e}")


if __name__ == "__main__":
    main()
