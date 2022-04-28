def get_paper_models():
    return [
        "efficientnet_b0",
        "resnet50", 
        "madry", 
        "levit_256",
        "levit_128s",
        "torchvision_resnet50",
        "tf_efficientnet_b0",
        "tf_efficientnet_b0_ap",
        "tf_efficientnetv2_b0",
        "tf_efficientnet_b0_ns",
        "mixnet_s",
        "resnetv2_50x1_bit_distilled",
        "vit_base_patch16_224",
        "mobilenetv2_140",
        "mobilenetv3_rw",
        "mobilenetv2_120d",
        "swsl_resnext101_32x4d",
        "mixnet_l",
        "twins_pcpvt_base",
        "coat_lite_small",
        "levit_384",
        "swsl_resnext50_32x4d",
        "pit_s_distilled_224",
        "rexnet_200",
        "rexnet_150",
        "rexnet_130",
        "convit_small",
        "swin_tiny_patch4_window7_224",
        "mixnet_xl",
        "dpn92",
        "dpn68b",
        "dla60x",
        "dla102",
        "regnetx_032",
        "hrnet_w30"
    ]


def models_to_variants(m):
    return [
                m,
                "RS-" + m + "-0.01-100",
                "RS-" + m + "-0.02-100",
                "RS-" + m + "-0.04-100",
                "RS-" + m + "-0.06-100",
                "RS-" + m + "-0.08-100",
                "RS-" + m + "-0.1-100",
                "FINETUNE-" + m + "-last_layer",
                "FINETUNE-" + m + "-all_model",
                "HALF-" + m,
                "HIST-" + m + "-equalize",
                "JPEG-" + m + "-90",
                "JPEG-" + m + "-80",
                "JPEG-" + m + "-70",
                "JPEG-" + m + "-60",
                "JPEG-" + m + "-50",
                "JPEG-" + m + "-40",
                "JPEG-" + m + "-30",
                "PRUNE-" + m + "-conv-0.1",
                "PRUNE-" + m + "-conv-0.2",
                "PRUNE-" + m + "-conv-0.3",
                "PRUNE-" + m + "-last-0.7",
                "PRUNE-" + m + "-last-0.8",
                "PRUNE-" + m + "-last-0.9",
                "PRUNE-" + m + "-last-0.95",
                "PRUNE-" + m + "-all-0.01",
                "PRUNE-" + m + "-all-0.02",
                "PRUNE-" + m + "-all-0.03",
                "PRUNE-" + m + "-all-0.04",
                "POSTERIZE-" + m + "-3",
                "POSTERIZE-" + m + "-4",
                "POSTERIZE-" + m + "-5",
                "POSTERIZE-" + m + "-6",
                "POSTERIZE-" + m + "-7"
            ]