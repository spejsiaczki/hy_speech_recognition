import os
import pathlib
import tempfile
from gunning_fog import GunningFog
from language_detection import LanguageDetection
from speech_processing import SpeechProcessing
from video_to_audio import VideoToAudio
from ocr import OCR


data_dir = pathlib.Path("./data")
ocr = OCR()
for filename in sorted(os.listdir(data_dir)):
    # Filename to process
    filename_abs = str((data_dir / filename).absolute())
    print("\033[92m" + filename_abs + "\033[0m")

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, dir='/tmp', suffix=".wav")
    temp_file_name = temp_file.name
    print("--------------------------")

    # Get subtitles

    subtitles = ocr.ocr(filename_abs)
    ocr_text = "".join(subtitles.values())
    for time, content in subtitles.items():
        print(f"[{time} s]: {content}")
    print("--------------------------")

    # Convert video to audio
    VideoToAudio(filename_abs, temp_file_name).run()
    print("--------------------------")

    # Create speech processing object
    speech_processing = SpeechProcessing(temp_file_name).run()
    print("--------------------------")

    # Print outputs
    print("\033[34m" + speech_processing.get_text() + "\033[0m")
    print("--------------------------")
    print(speech_processing.get_word_timestamps())
    print("--------------------------")
    print(speech_processing.get_pause_timestamps())

    # Language detection
    print("--------------------------")
    lang_detect = LanguageDetection()
    for word in speech_processing.get_words():
        print(word, lang_detect.detect_polish(word))

    # Gunning fog
    gf = GunningFog(speech_processing.get_text())
    print("--------------------------")
    print(gf.score)
    print("--------------------------")
    print(gf.grade_level_pl)
    print("--------------------------")

    # Clenup temp files
    os.remove(temp_file_name)
