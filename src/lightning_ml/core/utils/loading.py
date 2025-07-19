# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import glob
import os
import re
from functools import partial
from os import PathLike
from typing import Tuple, Union
from urllib.parse import parse_qs, quote, urlencode, urlparse

import fsspec
import numpy as np
import pandas as pd
import torch

from .imports import _TOPIC_AUDIO_AVAILABLE, _TORCHVISION_AVAILABLE, Image

if _TOPIC_AUDIO_AVAILABLE:
    from torchaudio.transforms import Spectrogram

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets.folder import IMG_EXTENSIONS
else:
    IMG_EXTENSIONS = (
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
    )

NP_EXTENSIONS = (".npy",)

AUDIO_EXTENSIONS = (
    ".aiff",
    ".au",
    ".avr",
    ".caf",
    ".flac",
    ".mat",
    ".mat4",
    ".mat5",
    ".mpc2k",
    ".ogg",
    ".paf",
    ".pvf",
    ".rf64",
    ".ircam",
    ".voc",
    ".w64",
    ".wav",
    ".nist",
    ".wavex",
)

CSV_EXTENSIONS = (".csv", ".txt")

TSV_EXTENSIONS = (".tsv",)

PATH_TYPE = Union[str, bytes, os.PathLike]


def has_file_allowed_extension(
    filename: PATH_TYPE, extensions: tuple[str, ...]
) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions

    """
    return str(filename).lower().endswith(extensions)


def _load_image_from_image(file):
    """Load an image file and convert it to RGB.

    Opens the image using PIL, ensures it is fully loaded, and converts to RGB mode.

    Args:
        file: A file-like object or path to the image.

    Returns:
        A PIL.Image.Image in RGB mode.
    """
    img = Image.open(file)
    img.load()
    return img.convert("RGB")


def _load_image_from_numpy(file):
    """Load a NumPy array as an RGB image.

    Reads a .npy file into a NumPy array of uint8 values, then converts to a PIL Image in RGB mode.

    Args:
        file: A file-like object or path to the .npy file.

    Returns:
        A PIL.Image.Image in RGB mode.
    """
    return Image.fromarray(np.load(file).astype("uint8")).convert("RGB")


def _load_spectrogram_from_image(file):
    """Load an image file as a float32 spectrogram array.

    Opens the image, converts it to RGB, and casts the pixel data to a float32 NumPy array.

    Args:
        file: A file-like object or path to the image.

    Returns:
        A NumPy array of shape (H, W, C) with dtype float32.
    """
    img = _load_image_from_image(file)
    return np.array(img).astype("float32")


def _load_spectrogram_from_numpy(file):
    """Load a NumPy array as a float32 spectrogram.

    Reads a .npy file into a NumPy array and casts values to float32.

    Args:
        file: A file-like object or path to the .npy file.

    Returns:
        A NumPy array with dtype float32.
    """
    return np.load(file).astype("float32")


def _load_spectrogram_from_audio(file, sampling_rate: int = 16000, n_fft: int = 400):
    """Compute a spectrogram from an audio file.

    Loads audio data at the specified sampling rate, computes a normalized spectrogram
    using torchaudio.transforms.Spectrogram, permutes dimensions to (Time, Frequency, Channels),
    and returns as a NumPy array.

    Args:
        file: A file-like object or path to the audio file.
        sampling_rate: Sampling rate for audio loading.
        n_fft: Number of FFT components for the spectrogram.

    Returns:
        A NumPy array representing the spectrogram with dtype float32.
    """
    # Import locally to prevent import errors if system dependencies are not available.
    import librosa
    from soundfile import SoundFile

    sound_file = SoundFile(file)
    waveform, _ = librosa.load(sound_file, sr=sampling_rate)
    return (
        Spectrogram(n_fft, normalized=True)(torch.from_numpy(waveform).unsqueeze(0))
        .permute(1, 2, 0)
        .numpy()
    )


def _load_audio_from_audio(file, sampling_rate: int = 16000):
    """Load raw audio waveform from a file.

    Reads the audio waveform at the specified sampling rate using librosa.

    Args:
        file: A file-like object or path to the audio file.
        sampling_rate: Sampling rate for audio loading.

    Returns:
        A NumPy array containing the audio waveform.
    """
    # Import locally to prevent import errors if system dependencies are not available.
    import librosa

    waveform, _ = librosa.load(file, sr=sampling_rate)
    return waveform


def _load_data_frame_from_csv(file, encoding: str):
    """Load a CSV file into a pandas DataFrame.

    Args:
        file: A file-like object or path to the CSV file.
        encoding: Character encoding to use when reading.

    Returns:
        A pandas.DataFrame containing the CSV data.
    """
    return pd.read_csv(file, encoding=encoding)


def _load_data_frame_from_tsv(file, encoding: str):
    """Load a TSV file into a pandas DataFrame.

    Args:
        file: A file-like object or path to the TSV file.
        encoding: Character encoding to use when reading.

    Returns:
        A pandas.DataFrame containing the TSV data.
    """
    return pd.read_csv(file, sep="\t", encoding=encoding)


_image_loaders = {
    IMG_EXTENSIONS: _load_image_from_image,
    NP_EXTENSIONS: _load_image_from_numpy,
}


_spectrogram_loaders = {
    IMG_EXTENSIONS: _load_spectrogram_from_image,
    NP_EXTENSIONS: _load_spectrogram_from_numpy,
    AUDIO_EXTENSIONS: _load_spectrogram_from_audio,
}


_audio_loaders = {
    AUDIO_EXTENSIONS: _load_audio_from_audio,
}


_data_frame_loaders = {
    CSV_EXTENSIONS: _load_data_frame_from_csv,
    TSV_EXTENSIONS: _load_data_frame_from_tsv,
}


def _get_loader(file_path: str, loaders):
    """Select the appropriate loader function based on file extension.

    Iterates through the provided mapping of extensions to loader functions and
    returns the first loader that matches the file's extension.

    Args:
        file_path: Path to the file whose loader is needed.
        loaders: Dict mapping tuples of extensions to loader callables.

    Returns:
        The loader callable corresponding to the file extension.

    Raises:
        ValueError: If no loader matches the file extension.
    """
    for extensions, loader in loaders.items():
        if has_file_allowed_extension(file_path, extensions):
            return loader
    raise ValueError(
        f"File: {file_path} has an unsupported extension. Supported extensions: "
        f"{list(sum(loaders.keys(), ()))}."
    )


WINDOWS_FILE_PATH_RE = re.compile("^[a-zA-Z]:(\\\\[^\\\\]|/[^/]).*")


def is_local_path(file_path: str) -> bool:
    """Determine if a path refers to a local filesystem path.

    Checks for Windows drive paths or file:// scheme to decide if the path is local.

    Args:
        file_path: The file path or URL to check.

    Returns:
        True if the path is local, False otherwise.
    """
    if WINDOWS_FILE_PATH_RE.fullmatch(file_path):
        return True
    return urlparse(file_path).scheme in ["", "file"]


def escape_url(url: str) -> str:
    """Escape and normalize a URL for safe usage.

    Quotes the path component and rebuilds the query string to ensure proper encoding.

    Args:
        url: The URL to escape.

    Returns:
        A properly escaped URL string.
    """
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{quote(parsed.path)}?{urlencode(parse_qs(parsed.query), doseq=True)}"


def escape_file_path(file_path: str | PathLike) -> str:
    """Escape a file system path or URL for fsspec opening.

    Uses glob.escape for local paths and escape_url for remote URLs.

    Args:
        file_path: The local path or URL to escape.

    Returns:
        An escaped string safe for fsspec.open.
    """
    file_path_str = str(file_path)
    return (
        glob.escape(file_path_str)
        if is_local_path(file_path_str)
        else escape_url(file_path_str)
    )


def load(file_path: str, loaders):
    """Load data from a file using the given loader mapping.

    Determines the correct loader via _get_loader, opens the file with fsspec,
    and applies the loader function to the file object.

    Args:
        file_path: Path or URL of the file to load.
        loaders: Mapping from file extensions to loader functions.

    Returns:
        The output of the loader function.
    """
    loader = _get_loader(file_path, loaders)
    # escaping file_path to avoid fsspec treating the path as a glob pattern
    # fsspec ignores `expand=False` in read mode
    with fsspec.open(escape_file_path(file_path)) as file:
        return loader(file)


def load_image(file_path: str):
    """Load an image from a file.

    Args:
        file_path: The image file to load.

    Returns:
        The loaded object (e.g., PIL.Image.Image, NumPy array, pandas.DataFrame).
    """
    return load(file_path, _image_loaders)


def load_spectrogram(file_path: str, sampling_rate: int = 16000, n_fft: int = 400):
    """Load a spectrogram from an image or audio file.

    Args:
        file_path: The file to load.
        sampling_rate: The sampling rate to resample to if loading from an audio file.
        n_fft: The size of the FFT to use when creating a spectrogram from an audio file.

    Returns:
        The loaded object (e.g., PIL.Image.Image, NumPy array, pandas.DataFrame).
    """
    loaders = copy.copy(_spectrogram_loaders)
    loaders[AUDIO_EXTENSIONS] = partial(
        loaders[AUDIO_EXTENSIONS], sampling_rate=sampling_rate, n_fft=n_fft
    )
    return load(file_path, loaders)


def load_audio(file_path: str, sampling_rate: int = 16000):
    """Load a waveform from an audio file.

    Args:
        file_path: The file to load.
        sampling_rate: The sampling rate to resample to.

    Returns:
        The loaded object (e.g., PIL.Image.Image, NumPy array, pandas.DataFrame).
    """
    loaders = {
        extensions: partial(loader, sampling_rate=sampling_rate)
        for extensions, loader in _audio_loaders.items()
    }
    return load(file_path, loaders)


def load_data_frame(file_path: str, encoding: str = "utf-8"):
    """Load a data frame from a CSV (or similar) file.

    Args:
        file_path: The file to load.
        encoding: The encoding to use when reading the file.

    Returns:
        The loaded object (e.g., PIL.Image.Image, NumPy array, pandas.DataFrame).
    """
    loaders = {
        extensions: partial(loader, encoding=encoding)
        for extensions, loader in _data_frame_loaders.items()
    }
    return load(file_path, loaders)
