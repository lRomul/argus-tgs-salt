# Resnet family definition dicts for help and as a reference

resnet18 = [
    {'type': 'basic', 'n_blocks': 2, 'n_filters': 64, 'stride': 1},
    {'type': 'basic', 'n_blocks': 2, 'n_filters': 128, 'stride': 2},
    {'type': 'basic', 'n_blocks': 2, 'n_filters': 256, 'stride': 2},
    {'type': 'basic', 'n_blocks': 2, 'n_filters': 512, 'stride': 2},
]

resnet34 = [
    {'type': 'basic', 'n_blocks': 3, 'n_filters': 64, 'stride': 1},
    {'type': 'basic', 'n_blocks': 4, 'n_filters': 128, 'stride': 2},
    {'type': 'basic', 'n_blocks': 6, 'n_filters': 256, 'stride': 2},
    {'type': 'basic', 'n_blocks': 3, 'n_filters': 512, 'stride': 2},
]

resnet50 = [
    {'type': 'bottleneck', 'n_blocks': 3, 'n_filters': 64, 'stride': 1},
    {'type': 'bottleneck', 'n_blocks': 4, 'n_filters': 128, 'stride': 2},
    {'type': 'bottleneck', 'n_blocks': 6, 'n_filters': 256, 'stride': 2},
    {'type': 'bottleneck', 'n_blocks': 3, 'n_filters': 512, 'stride': 2},
]

resnet101 = [
    {'type': 'bottleneck', 'n_blocks': 3, 'n_filters': 64, 'stride': 1},
    {'type': 'bottleneck', 'n_blocks': 4, 'n_filters': 128, 'stride': 2},
    {'type': 'bottleneck', 'n_blocks': 23, 'n_filters': 256, 'stride': 2},
    {'type': 'bottleneck', 'n_blocks': 3, 'n_filters': 512, 'stride': 2},
]

resnet152 = [
    {'type': 'bottleneck', 'n_blocks': 3, 'n_filters': 64, 'stride': 1},
    {'type': 'bottleneck', 'n_blocks': 8, 'n_filters': 128, 'stride': 2},
    {'type': 'bottleneck', 'n_blocks': 36, 'n_filters': 256, 'stride': 2},
    {'type': 'bottleneck', 'n_blocks': 3, 'n_filters': 512, 'stride': 2},
]
