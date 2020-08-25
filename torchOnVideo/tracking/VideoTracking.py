class VideoTracking():
    def __init__(self, tracking_type):
        self.TrackingType = tracking_type

    def get_models(self):
        raise NotImplementedError
