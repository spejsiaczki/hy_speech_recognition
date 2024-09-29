import logging
import ffmpeg

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(asctime)s][%(levelname)s][%(name)s]: %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class VideoToAudio:
    def __init__(self,
                 input_path: str,
                 output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        logger.debug(self.__class__.__name__ + " initialized")

    def run(self) -> "VideoToAudio":
        logger.debug(
            f"Running video to audio conversion from {self.input_path} to {self.output_path}")
        self._preprocess_audio(self.input_path, self.output_path)
        return self

    def _preprocess_audio(self,
                          input_file: str,
                          output_file: str) -> None:
        try:
            ffmpeg \
                .input(input_file) \
                .output(output_file,
                        acodec='pcm_s16le',
                        ar=44100,
                        ) \
                .overwrite_output() \
                .run(quiet=True)

            logger.debug(
                f"Audio extracted from {input_file}"
                f"and saved to {output_file}")
        except ffmpeg.Error:
            logger.exception("Error occurred while extracting audio")
