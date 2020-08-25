from ..SuperResolution import SuperResolution

## subclass of SuperResolution class
class SOF_VSR(SuperResolution):
    def __init__(self, scale):
        super(SOF_VSR, self).__init__(scale=scale)
