import torch
import pathlib
from typing import Tuple
from PIL import Image
from torchvision.transforms import v2, InterpolationMode
from torch.utils.data import Dataset


class TrainDIV2K(Dataset):

    def __init__(self, dir_path: str, scale: int = 3):
        super().__init__()
        self.paths = list(pathlib.Path(dir_path).glob("*.png"))
        self.scale = scale

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.paths[idx]).convert('RGB')
        return self._transform(image)

    def _transform(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        h = (image.height // self.scale) * self.scale
        w = (image.width // self.scale) * self.scale

        to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        lr = v2.Compose([
            v2.Resize((h // self.scale, w // self.scale), interpolation=InterpolationMode.BICUBIC),
            to_tensor,
        ])(image)

        hr = v2.Compose([
            v2.Resize((h, w), interpolation=InterpolationMode.BICUBIC),
            to_tensor,
        ])(image)

        return lr, hr


class EvalDIV2K(Dataset):

    def __init__(self, dir_path: str, scale: int = 3):
        super().__init__()
        self.paths = list(pathlib.Path(dir_path).glob("*.png"))
        self.scale = scale

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.paths[idx]).convert('RGB')
        return self._transform(image)

    def _transform(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        h = (image.height // self.scale) * self.scale
        w = (image.width // self.scale) * self.scale

        to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        lr = v2.Compose([
            v2.Resize((h // self.scale, w // self.scale), interpolation=InterpolationMode.BICUBIC),
            to_tensor,
        ])(image)

        hr = v2.Compose([
            v2.Resize((h, w), interpolation=InterpolationMode.BICUBIC),
            to_tensor,
        ])(image)

        return lr, hr
