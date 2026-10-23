"""Real-weight variable-grid probe; standalone research, no runtime changes."""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def explicit_gelu(x):
    import torch
    return (x*.5)*(1+torch.tanh((x+(x*x*x)*.044715)*.7978845608028654))

def make_graph(pack, capacity=16384, query_tile=1024, depth=27, fuse_position=False, norm_mode="safe"):
    print(json.dumps({"stage":"build_import_begin","time":time.time()}),flush=True)
    import torch
    import torch.nn.functional as F
    import coreai_torch
    import vision_weights as R
    torch.set_num_threads(4)
    layout, weights = R.load_pack(pack)
    from PIL import Image
    import math

    def conv(name):
        weight = weights[name + '.weight']
        module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], 1)
        with torch.no_grad():
            module.weight.copy_(torch.from_numpy(weight[:, :, None, None]))
            module.bias.copy_(torch.from_numpy(weights[name + '.bias']))
        return module

    class Norm(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.range_scale = 8 if name.startswith('blocks.') else 1152**.5
            gain = 1152**.5 / self.range_scale
            self.register_buffer('weight', torch.from_numpy(weights[name + '.weight'].copy() * gain)[None, :, None, None])
            self.register_buffer('bias', torch.from_numpy(weights[name + '.bias'].copy())[None, :, None, None])

        def forward(self, x):
            center = x - x.mean(1, keepdim=True)
            scaled = center * (1 / self.range_scale)
            variance = (scaled * scaled).sum(1, keepdim=True)
            epsilon = 1e-6 * 1152 / (self.range_scale*self.range_scale)
            return center * torch.rsqrt(variance + epsilon) * self.weight + self.bias

    class Block(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            prefix = f'blocks.{index}.'
            self.n1 = Norm(prefix + 'norm1')
            self.n2 = Norm(prefix + 'norm2')
            self.qkv = conv(prefix + 'qkv')
            self.proj = conv(prefix + 'proj')
            self.f1 = conv(prefix + 'fc1')
            self.f2 = conv(prefix + 'fc2')

        def forward(self, x, cosine, sine, key_mask):
            qs, ks, vs = self.qkv(self.n1(x)).split(1152, 1)

            def rope(z):
                return z * cosine + torch.cat((-z[:, 36:], z[:, :36]), 1) * sine

            contexts = []
            for head in range(16):
                q = rope(qs[:, head * 72:(head + 1) * 72])
                k = rope(ks[:, head * 72:(head + 1) * 72])
                v = vs[:, head * 72:(head + 1) * 72]
                tiles = []
                for start in range(0, capacity, query_tile):
                    scores = torch.einsum('bchq,bkhc->bkhq', q[..., start:start + query_tile], k.transpose(1, 3)) * (72 ** -.5)
                    probabilities = torch.softmax(scores + key_mask, dim=1)
                    # Exact power-of-two range shift; avoids the measured tiny-P PV loss.
                    tiles.append(torch.einsum('bkhq,bchk->bchq', probabilities * 1024, v) / 1024)
                contexts.append(torch.cat(tiles, dim=3))
            x = x + self.proj(torch.cat(contexts, 1))
            return x + self.f2(explicit_gelu(self.f1(self.n2(x))))

    class Graph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch = conv('patch_embed')
            if fuse_position:
                fused = torch.nn.Conv2d(1536 + 1152, 1152, 1)
                with torch.no_grad():
                    fused.weight[:, :1536].copy_(self.patch.weight)
                    fused.weight[:, 1536:].copy_(torch.eye(1152)[:, :, None, None])
                    fused.bias.copy_(self.patch.bias)
                self.patch = fused
            self.blocks = torch.nn.ModuleList([Block(index) for index in range(depth)])
            self.norm = Norm('merger.norm')
            self.merge1 = conv('merger.fc1')
            self.merge2 = conv('merger.fc2')

        def forward(self, patches, position, cosine, sine, key_mask):
            if fuse_position:
                x = self.patch(torch.cat((patches.T[None, :, None, :], position), dim=1))
            else:
                x = self.patch(patches.T[None, :, None, :]) + position
            for block in self.blocks:
                x = block(x, cosine, sine, key_mask)
            merged = self.norm(x)[0, :, 0, :].T.reshape(capacity // 4, 4608).T[None, :, None, :]
            return self.merge2(explicit_gelu(self.merge1(merged)))[0, :, 0, :].T

    return Graph().eval()
