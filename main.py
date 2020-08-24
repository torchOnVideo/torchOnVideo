# <<<<<<< HEAD
from torchOnVideo.datasets import f16_video_dataset



print(f16_video_dataset.__all__)

f16_obj = f16_video_dataset.__dict__['denoising'](5)
f16_obj.run_f(10)
# =======
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
# >>>>>>> aa908c02348f03b019611579567f4cb990640b31

# from torchOnVideo.frame_interpolation.models import CAIN
from torchOnVideo.datasets.Vimeo90KTriplet.frame_interpolation.train_adacof import TrainAdaCoF