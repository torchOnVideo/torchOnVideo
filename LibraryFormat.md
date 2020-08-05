# Library Format

For example SuperResolution

Say we are using the *ISeeBetter* model:

    tov_obj = torchOnVideo.SuperResolution.ISeeBetter()
    
For Training:

    tov_obj.train()  # All dataset information and parameters if needed

For inference: Those who want to use a pretrained model

     tov_obj.infer() # pretrained_model_location, stored_video_location/frames if needed; Parameters and information needed
     
*Additionally please keep on focusing on the datasets, samplers, etc - 
If there is any specific functionality required it will go in one of our common submodules such as:
* torchOnVideo.datasets
* torchOnVideo.samplers
* torchOnVideo.transforms
* torchOnVideo.tools


