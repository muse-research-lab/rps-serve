from dataclasses import dataclass
from typing import Union, LiteralString

import numpy.typing as npt
import librosa


@dataclass(frozen=True)
class AudioAsset:
    path: Union[str,LiteralString]

    @property
    def data(self) -> npt.NDArray:
        return librosa.load(self.path, sr=None)[0]