import torchOnVideo
from torchOnVideo.super_resolution.models import DBPNS
import torchOnVideo.losses

base_filter = 256
feat = 64
num_stages = 3
scale_factor = 4
DBPNS(base_filter, feat, num_stages, scale_factor)



from torchOnVideo.super_resolution import SOF_VSR

from torchOnVideo.super_resolution.SOF_VSR import TrainModel
SOF_VSR_OBJ = SOF_VSR(scale=4)

SOF_VSR_OBJ.run_models()
# SOF_VSR_OBJ.get_models()


tm_obj = TrainModel(scale=4)
tm_obj()
