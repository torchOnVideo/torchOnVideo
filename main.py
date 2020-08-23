from torchOnVideo.datasets import f16_video_dataset



print(f16_video_dataset.__all__)

f16_obj = f16_video_dataset.__dict__['denoising'](5)
f16_obj.run_f(10)