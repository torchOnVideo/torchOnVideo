from ..SuperResolution import SuperResolution

## subclass of SuperResolution class
class ISeeBetter(SuperResolution):
    def __init__(self, scale):
        super(ISeeBetter, self).__init__(scale=scale)

