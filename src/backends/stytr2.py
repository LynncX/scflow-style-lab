"""Pinned StyTr2 with a versioned inference-only/device portability patch."""
import importlib
import sys
from types import SimpleNamespace

from .torch_image import TorchImageBackend


class StyTr2Backend(TorchImageBackend):
    method = "stytr2"

    def build(self, torch):
        sys.path.insert(0, str(self.source))
        model = importlib.import_module("models.StyTR")
        transformer = importlib.import_module("models.transformer")
        vgg = model.vgg
        vgg.load_state_dict(self.weight(torch, "vgg"))
        vgg = torch.nn.Sequential(*list(vgg.children())[:44])
        decoder = model.decoder
        trans = transformer.Transformer()
        embedding = model.PatchEmbed()
        for key, module in (("decoder", decoder), ("transformer", trans), ("embedding", embedding)):
            module.load_state_dict(self.weight(torch, key))
        self.network = model.StyTrans(vgg, decoder, embedding, trans,
                                     SimpleNamespace(position_embedding="sine", hidden_dim=512))
        self.network.eval().to(self.device)

    def forward(self, content, style):
        return self.network(content, style)
