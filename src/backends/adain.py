"""Thin single-reference AdaIN adapter using the complete pinned upstream checkout."""
import importlib
import sys

from .torch_image import TorchImageBackend


class AdaINBackend(TorchImageBackend):
    method = "adain"

    def build(self, torch):
        sys.path.insert(0, str(self.source))
        net = importlib.import_module("net")
        self.adain = importlib.import_module("function").adaptive_instance_normalization
        self.decoder = net.decoder
        self.decoder.load_state_dict(self.weight(torch, "decoder"))
        net.vgg.load_state_dict(self.weight(torch, "vgg"))
        self.vgg = torch.nn.Sequential(*list(net.vgg.children())[:31])
        self.decoder.eval().to(self.device)
        self.vgg.eval().to(self.device)
        self.alpha = float(self.options.get("alpha", 1.0))
        if not 0 <= self.alpha <= 1:
            raise ValueError("AdaIN alpha must be between 0 and 1")

    def forward(self, content, style):
        content_f, style_f = self.vgg(content), self.vgg(style)
        feature = self.adain(content_f, style_f)
        return self.decoder(feature * self.alpha + content_f * (1 - self.alpha))
