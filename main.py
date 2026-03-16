"""
Video Transcription with Speaker Diarization

Extracts audio from video files and produces speaker-labeled transcripts
using WhisperX (Whisper + PyAnnote). All processing runs locally.
"""

import os
import gc
import subprocess
import warnings

import torch
import whisperx
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline
from datetime import timedelta
from datetime import timedelta

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


def extract_audio(video_path: str) -> str:
    """Extract audio from a video file using ffmpeg and save as WAV."""
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    command = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
    #    "-af", "highpass=f=80, afftdn=nr=12:nf=-50:tn=1, loudnorm",
        "-ar", "44100", "-ac", "2",
        audio_path, "-y"
    ]
    subprocess.run(command, check=True)
    return audio_path

# def ffmpef_clean_audio(input_path: str, output_path: str):
#     """Clean audio using ffmpeg filters."""
#     command = [
#         "ffmpeg", "-y", "-i" input_path,
#         "-af", "highpass=f=80, afftdn=nr=12:nf-50:tn=1, loudnorm",
#         "-ac", "1", "-ar", "16000",
#         output_path, "-y"
#     ]
#     subprocess.run(command, check=True)


    
def transcribe_and_diarize(audio_path: str, hf_token: str) -> list[dict]:
    """
    Run the full WhisperX pipeline:
      1. Transcribe with Whisper
      2. Align word-level timestamps
      3. Diarize speakers with PyAnnote
      4. Assign speaker labels to segments
    """
    device = "cpu"  # Safest default for Mac; change to "cuda" on GPU machines
    compute_type = "int8"
    batch_size = 16

    # --- Step 1: Transcribe ---
    print("1. Transcribing audio...")
    model = whisperx.load_model("turbo", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)

    gc.collect()

    # --- Step 2: Align timestamps ---
    print("2. Aligning timestamps...")
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False,
    )

    gc.collect()

    # --- Step 3: Speaker diarization ---
    print("3. Performing speaker diarization...")
    diarize_model = DiarizationPipeline(
        use_auth_token=hf_token, device=device
    )
    diarize_segments = diarize_model(audio)

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


def main():
    load_dotenv(override=True)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN not set.")
        print("Create a .env file with: HF_TOKEN=your_token_here")
        return

    video_path = "data/M.AE2S5Q.02.12.25.webm"
    audio_path = extract_audio(video_path)
    # segments = transcribe_and_diarize(audio_path, hf_token)

    # print("\n--- Transcription ---\n")
    # for seg in segments:
    #     speaker = seg.get("speaker", "UNKNOWN")
    #     text = seg.get("text", "").strip()
    #     start = seg.get("start", seg.get("start_time", 0.0))
    #     ts = format_timestamp(start)
    #     print(f"[{ts}] [{speaker}]: {text}")


if __name__ == "__main__":
    main()
