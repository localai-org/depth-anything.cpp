#!/usr/bin/env python3
"""Convert the yuvraj108c Depth Anything V2 ONNX exports to GGUF.

The legacy depth-upscale service uses ONNX Runtime directly. depth-anything.cpp
uses the same parameter tensors through its DA2 GGUF route, so this converter
copies ONNX initializers into the existing tensor/metadata schema without
requiring PyTorch or the upstream Depth Anything source tree.
"""
import argparse
import math
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gguf
import scripts.gguf_keys as K


ENCODERS = {
    "vits": (384, 12, 6, [2, 5, 8, 11]),
    "vitb": (768, 12, 12, [2, 5, 8, 11]),
    "vitl": (1024, 24, 16, [4, 11, 17, 23]),
    "vitg": (1536, 40, 24, [9, 19, 29, 39]),
}


def infer_encoder(initializers):
    patch = initializers.get("pretrained.patch_embed.proj.weight")
    if patch is None or len(patch.dims) != 4 or patch.dims[1:] != [3, 14, 14]:
        raise ValueError("expected a Depth Anything V2 [embed,3,14,14] patch embedding")
    embed_dim = int(patch.dims[0])
    block_count = 0
    while f"pretrained.blocks.{block_count}.attn.qkv.weight" in initializers:
        block_count += 1
    for name, (expected_embed, expected_blocks, _, _) in ENCODERS.items():
        if (embed_dim, block_count) == (expected_embed, expected_blocks):
            return name
    raise ValueError(f"unsupported DA2 backbone: embed_dim={embed_dim}, blocks={block_count}")


def inferred_max_depth(path):
    name = os.path.basename(path).lower()
    if "metric_hypersim" in name:
        return 20.0
    if "metric_vkitti" in name:
        return 80.0
    return 0.0


def validate_io(model):
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("expected one ONNX input and one output")
    def dims(value):
        return [int(d.dim_value) for d in value.type.tensor_type.shape.dim]
    if dims(model.graph.input[0]) != [1, 3, 518, 518]:
        raise ValueError(f"expected input [1,3,518,518], got {dims(model.graph.input[0])}")
    if dims(model.graph.output[0]) != [1, 518, 518]:
        raise ValueError(f"expected output [1,518,518], got {dims(model.graph.output[0])}")


def write_da2_onnx_gguf(onnx_path, output, checkpoint_name, max_depth):
    model = onnx.load(onnx_path, load_external_data=False)
    validate_io(model)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    encoder = infer_encoder(initializers)
    embed_dim, depth, num_heads, out_layers = ENCODERS[encoder]
    pos_embed = initializers.get("pretrained.pos_embed")
    if pos_embed is None or list(pos_embed.dims) != [1, 1369 + 1, embed_dim]:
        raise ValueError("invalid DA2 positional embedding")
    head_features = int(initializers["depth_head.scratch.layer1_rn.weight"].dims[0])
    head_out_channels = [int(initializers[f"depth_head.projects.{i}.weight"].dims[0]) for i in range(4)]
    qkv_bias = "pretrained.blocks.0.attn.qkv.bias" in initializers
    fc1 = initializers.get("pretrained.blocks.0.mlp.fc1.weight")
    w12 = initializers.get("pretrained.blocks.0.mlp.w12.weight")
    if fc1 is not None:
        mlp_hidden = int(fc1.dims[0])
        ffn_type = "mlp"
    elif w12 is not None:
        mlp_hidden = int(w12.dims[0]) // 2
        ffn_type = "swiglu"
    else:
        raise ValueError("unsupported DA2 MLP tensors")
    pos_grid = math.isqrt(int(pos_embed.dims[1]) - 1)

    writer = gguf.GGUFWriter(output, K.ARCH)
    writer.add_string(K.KV["arch"], "depthanything2")
    writer.add_string(K.KV["checkpoint_name"], checkpoint_name)
    writer.add_uint32(K.KV["patch_size"], 14)
    writer.add_uint32(K.KV["vit.embed_dim"], embed_dim)
    writer.add_uint32(K.KV["vit.depth"], depth)
    writer.add_uint32(K.KV["vit.num_heads"], num_heads)
    writer.add_uint32(K.KV["vit.head_dim"], embed_dim // num_heads)
    writer.add_uint32(K.KV["vit.mlp_hidden"], mlp_hidden)
    writer.add_string(K.KV["vit.ffn_type"], ffn_type)
    writer.add_uint32(K.KV["vit.num_register"], 0)
    writer.add_float32(K.KV["vit.init_values"], 1.0)
    writer.add_int32(K.KV["vit.alt_start"], -1)
    writer.add_int32(K.KV["vit.rope_start"], -1)
    writer.add_int32(K.KV["vit.qknorm_start"], -1)
    writer.add_float32(K.KV["vit.rope_freq"], 100.0)
    writer.add_bool(K.KV["vit.cat_token"], False)
    writer.add_bool(K.KV["vit.qkv_bias"], qkv_bias)
    writer.add_float32(K.KV["vit.ln_eps"], 1e-6)
    writer.add_float32(K.KV["vit.interp_offset"], 0.1)
    writer.add_bool(K.KV["vit.interp_antialias"], False)
    writer.add_uint32(K.KV["vit.pos_embed_grid"], pos_grid)
    writer.add_array(K.KV["vit.out_layers"], out_layers)
    writer.add_array(K.KV["img.mean"], [0.485, 0.456, 0.406])
    writer.add_array(K.KV["img.std"], [0.229, 0.224, 0.225])
    writer.add_string(K.KV["img.resize_mode"], "lower_bound")
    writer.add_uint32(K.KV["img.resize_target"], 518)
    writer.add_uint32(K.KV["head.features"], head_features)
    writer.add_array(K.KV["head.out_channels"], head_out_channels)
    writer.add_uint32(K.KV["head.output_dim"], 1)
    writer.add_bool(K.KV["head.pos_embed"], False)
    writer.add_string(K.KV["head.activation"], "sigmoid" if max_depth > 0 else "relu")
    writer.add_string(K.KV["head.norm_type"], "idt")
    if max_depth > 0:
        writer.add_float32(K.KV["head.max_depth"], max_depth)

    written = 0
    for name, tensor in initializers.items():
        if name.startswith("pretrained."):
            gguf_name = K.rename_backbone(name.removeprefix("pretrained."))
        elif name.startswith("depth_head."):
            gguf_name = K.rename_head(name.removeprefix("depth_head."))
        else:
            raise ValueError(f"unexpected ONNX initializer {name}")
        if gguf_name is None:
            raise ValueError(f"unmapped ONNX initializer {name}")
        values = np.ascontiguousarray(numpy_helper.to_array(tensor), dtype=np.float32)
        writer.add_tensor(gguf_name, values)
        written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {output}: tensors={written} arch=depthanything2 encoder={encoder} max_depth={max_depth:g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Depth Anything V2 ONNX model")
    parser.add_argument("--output", required=True, help="destination GGUF path")
    parser.add_argument("--name", default="yuvraj108c/Depth-Anything-2-Onnx")
    parser.add_argument("--max-depth", type=float, default=None,
                        help="metric range in metres; inferred from metric model filenames")
    args = parser.parse_args()
    max_depth = inferred_max_depth(args.onnx) if args.max_depth is None else args.max_depth
    write_da2_onnx_gguf(args.onnx, args.output, args.name, max_depth)


if __name__ == "__main__":
    main()
