from dataclasses import dataclass, field
from typing import LiteralString, Union

from PIL import Image

@dataclass
class ImageAsset:
    path: Union[str,LiteralString]
    image: Image.Image = field(init=False, repr=False)

    def __post_init__(self):
        self.image = Image.open(self.path).convert("RGB")
    
    @property
    def size(self) -> tuple[int, int]:
        return self.image.size
    
    @property
    def data(self) -> Image.Image:
        return self.image
    
    def compress_image(self, width: int, height: int):
        self.image = self.image.resize(
            (width, height), resample=Image.Resampling.LANCZOS
        )