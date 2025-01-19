from methods.byol import BYOL, TransformBYOL
from methods.dino import DINO, TransformDINO

METHODS = {
    "byol": (BYOL, TransformBYOL),
    "dino": (DINO, TransformDINO),
}
