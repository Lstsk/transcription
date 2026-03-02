import whisper
import subprocess


class ExtractAudio:
    """
    A class to extract audio from a video file using ffmpeg.    
    """
    def __init__(self, video_path):
        # No space in video path
        self.video_path = video_path

    def extract_audio(self):
        # Use ffmpeg to extract audio from the video, don't edit the video, just extract the audio and save it as a wav file'
        audio_path = self.video_path.split(".")[0] + ".wav"
        command = f"ffmpeg -i {self.video_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_path}"
        subprocess.run(command, shell=True)
        return audio_path


class TranscribeAudio:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.model = whisper.load_model("turbo")
    
    def transcribe(self):
        result = self.model.transcribe(self.audio_path)
        return result["text"]


def main(): 
    # Extract audio from the video file
    # video_path = "cleaned.wav" 
    # extractor = ExtractAudio(video_path)
    # audio_path = extractor.extract_audio()
    
    transcriber= TranscribeAudio("cleaned.wav")
    transcription = transcriber.transcribe()
    print(transcription)

if __name__ == "__main__":
    main()
