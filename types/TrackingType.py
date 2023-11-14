import numpy as np

class TrackingData(object):
    def __init__(self, track_name, data):
        self.track_name = track_name
        self.data = data
        assert isinstance(self.data, np.ndarray), "Data should be a numpy array"
        assert self.data.ndim == 2, "Data should be a 2D numpy array"
        assert self.data[0].shape > 5, "Data should have at least 6 columns with same order with MOT17 format"