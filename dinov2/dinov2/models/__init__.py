# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import torch
from . import vision_transformer as vits


logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        #state_dict = torch.load('/ruiyan/yuhao/project/FMBC/dinov2/dinov2/models/UNI_updated.bin')
        # student.load_state_dict(state_dict, strict=False)  # `strict=False` 允许不完全匹配的加载
        # teacher.load_state_dict(state_dict, strict=False)
        initial_dict = torch.load('/ruiyan/yuhao/project/FMBC/dinov2/dinov2/models/UNI-1.5-updated.pth')
        student.load_state_dict(initial_dict)
        teacher.load_state_dict(initial_dict)
        with open ('/ruiyan/yuhao/project/FMBC/dinov2/dinov2/models/load.txt','w') as f:
            f.write('load successfully')
        print("Load pretrain weights successfully")

        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)
