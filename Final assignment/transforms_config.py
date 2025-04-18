from torchvision.transforms.v2 import Compose, Resize, Normalize, ToImage, ToDtype
import torch

TRANSFORM_CONFIG = {
    "afformer-tiny": Compose([
        ToImage(),
        Resize((512, 512)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ]),
    "afformer-tiny-batch32": Compose([
        ToImage(),
        Resize((512, 512)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ]),
    "attention--unet-training": Compose([
        ToImage(),
        Resize((512, 512)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ]),
    "attention--unet-training-pretrained-end-with-conv": Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ]),
    "attention--unet-training-pretrained-end-with-conv-scheduler": Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ]),
    "attention--unet-training-pretrained-end-with-conv-scheduler-freeze-and-unfreeze-128batch-512px": Compose([
        ToImage(),
        Resize((512, 512)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ]),
    "attention--unet-training-pretrained-end-with-conv-scheduler-freeze-and-unfreeze-64batch-512px": Compose([
        ToImage(),
        Resize((512, 512)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ]),
    "BowlNet_64_batch_512px": Compose([
        ToImage(),
        Resize((512, 512)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ]),
    "unet-training": Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ]),
    # add more per folder/model name
}