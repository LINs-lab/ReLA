import torch
import torch.nn as nn
import timm


def load_model_weights(model, model_path):
    state = torch.load(model_path, map_location="cpu")
    for key in model.state_dict():
        if "num_batches_tracked" in key:
            continue
        p = model.state_dict()[key]
        if key in state["state_dict"]:
            ip = state["state_dict"][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print(
                    "could not load layer: {}, mismatch shape {} ,{}".format(
                        key, (p.shape), (ip.shape)
                    )
                )
        else:
            print("could not load layer: {}, not in checkpoint".format(key))
    return model


def load_solo_model(ckpt_path, backbone, prune=True, resnet=True):

    if resnet:
        backbone.fc = nn.Identity()
        if prune:
            backbone.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=2, bias=False
            )
            backbone.maxpool = nn.Identity()

    assert (
        ckpt_path.endswith(".ckpt")
        or ckpt_path.endswith(".pth")
        or ckpt_path.endswith(".pt")
    )

    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            state[k.replace("encoder", "backbone")] = state[k]
            print(
                "You are using an older checkpoint. Use a new one as some issues might arrise."
            )
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]
    backbone.load_state_dict(state, strict=False)

    return backbone


class CLIP_RE(nn.Module):
    def __init__(self, model):
        super(CLIP_RE, self).__init__()

        self.net = model

    def forward(self, x):
        f = self.net.encode_image(x)
        return f


prior_input_size_dict = {
    "cf10": 32,
    "cf100": 32,
    "tin": 64,
    "in1k": 224,
    "in21k": 224,
    "in21k_mb": 224,
    "clip": 224,
    "clip_t4": 288,
    "clip_101": 224,
    "clip_b32": 224,
    "clip_b16": 224,
    "clip_l14": 224,
    "yolov9": 640,
}


def get_prior_model(name, *args):
    if name == "cf10":
        model = torch.load(
            "boostor/priors/cf10/byol_128_100_resnet18_modified_32_CIFAR10.pt",
            map_location="cpu",
        )

    elif name == "cf100":
        model = torch.load(
            "boostor/priors/cf100/byol_128_100_resnet18_modified_32_CIFAR100.pt",
            map_location="cpu",
        )

    elif name == "tin":
        model = torch.load(
            "boostor/priors/tin/byol_128_100_resnet18_modified_64_TinyImageNet.pt",
            map_location="cpu",
        )

    elif name == "in1k":
        model = torch.load(
            "boostor/priors/in1k/byol_512_100_resnet50_224_ImageNet-1K.pt",
            map_location="cpu",
        )

    elif name == "in21k":
        model = timm.create_model("resnet50", pretrained=False)
        model.fc = nn.Identity()
        model = load_model_weights(model, "boostor/priors/in21k/resnet50_miil_21k.pth")

    elif name == "in21k_mb":
        model = timm.create_model("mobilenetv3_large_100_miil", pretrained=False)
        model.classifier = nn.Identity()
        model = load_model_weights(
            model, "boostor/priors/in21k/mobilenetv3_large_100_miil_21k.pth"
        )

    elif name == "clip":
        model = torch.jit.load("boostor/priors/clip/RN50.pt", "cpu")
        model = CLIP_RE(model)

    elif name == "clip_t4":
        model = torch.jit.load("boostor/priors/clip/RN50x4.pt", "cpu")
        model = CLIP_RE(model)

    elif name == "clip_101":
        model = torch.jit.load("boostor/priors/clip/RN101.pt", "cpu")
        model = CLIP_RE(model)

    elif name == "clip_b16":
        model = torch.jit.load("boostor/priors/clip/ViT-B-16.pt", "cpu")
        model = CLIP_RE(model)

    elif name == "clip_b32":
        model = torch.jit.load("boostor/priors/clip/ViT-B-32.pt", "cpu")
        model = CLIP_RE(model)

    elif name == "clip_l14":
        model = torch.jit.load("boostor/priors/clip/ViT-L-14.pt", "cpu")
        model = CLIP_RE(model)

    elif name == "yolov9":
        model = torch.jit.load("boostor/priors/yolov9/yolov9_traced_model.pt", "cpu")

    return model
