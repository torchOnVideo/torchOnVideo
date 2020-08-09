import torchOnVideo
from torchOnVideo.super_resolution.models import DBPNS
import torchOnVideo.losses

base_filter = 256
feat = 64
num_stages = 3
scale_factor = 4
DBPNS(base_filter, feat, num_stages, scale_factor)


