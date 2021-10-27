from .build import build_discriminator, build_generator


# from .autogan_cifar10_a import AutoGANCIFAR10ADiscriminator, AutoGANCIFAR10AGenerator
from .path_aware_resnet_generator import PathAwareResNetGen, PathAwareResNetGenCBN
from .biggan_gen_disc import BigGANDisc
# from . import dense_generator
# from . import stylegan_v2