from .styleunet.styleunet_2 import StyleUNet
from .styleunet.styleunet_2 import SimpleUNet
from .spade_decoder import SPADEDecoder
Nueral_Refiner_Model={
    'styleunet':StyleUNet,
    'simpleunet':SimpleUNet,
    'spade':SPADEDecoder,}