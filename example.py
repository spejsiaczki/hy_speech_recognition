import os
import pathlib
from gunning_fog import GunningFog
from language_detection import LanguageDetection
from speech_processing import SpeechProcessing
from video_to_audio import VideoToAudio
from ocr import OCR
from text_comparison import TextComparison


data_dir = pathlib.Path("./data")
ocr = OCR()
for filename in sorted(os.listdir(data_dir)):

    # Filename to process
    filename_abs = str((data_dir / filename).absolute())
    print("\033[92m" + filename_abs + "\033[0m")

    # Create a temporary file
    temp_file_name = "/tmp/temp_audio.wav"
    print("--------------------------")

    # Convert video to audio
    VideoToAudio(filename_abs, temp_file_name).run()
    print("--------------------------")

    # Create speech processing object
    speech_processing = SpeechProcessing(temp_file_name).run()
    print("--------------------------")

    # Get subtitles
    subtitles = ocr.ocr(filename_abs)
    ocr_text = "".join(subtitles.values())
    for time, content in subtitles.items():
        print(f"[{time} s]: {content}")
    print("\033[33m" + ocr_text + "\033[0m")
    print("--------------------------")

    # Print outputs
    print("\033[34m" + speech_processing.get_text() + "\033[0m")
    print("--------------------------")

    # Print comparison
    text_comparison = TextComparison()
    print("leven:", text_comparison.compare_leven(
        ocr_text, speech_processing.get_text()))
    print("length:", text_comparison.compare_length(
        ocr_text, speech_processing.get_text()))
    print("total:", text_comparison.compare(
        ocr_text, speech_processing.get_text()))
    print("description:", text_comparison.compare_with_description(
        ocr_text, speech_processing.get_text()))
    print("--------------------------")

    # # Print timestamps
    # print(speech_processing.get_word_timestamps())
    # print("--------------------------")
    # print(speech_processing.get_pause_timestamps())

    # # Language detection
    # print("--------------------------")
    # lang_detect = LanguageDetection()
    # for word in speech_processing.get_words():
    #     print(word, lang_detect.detect_polish(word))

    # # Gunning fog
    # gf = GunningFog(speech_processing.get_text())
    # print("--------------------------")
    # print(gf.score)
    # print("--------------------------")
    # print(gf.grade_level_pl)
    # print("--------------------------")

    # Clenup temp files
    os.remove(temp_file_name)
