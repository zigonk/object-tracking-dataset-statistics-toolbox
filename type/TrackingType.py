import numpy as np


class TrackingData(object):
    def __init__(self, track_name, data):
        self.track_name = track_name
        self.data = data
        if (len(self.data) == 0):
            print("Warning: Track {} has no data".format(track_name))
            self.data = np.zeros((0, 6))
            return
        assert isinstance(
            self.data, np.ndarray), "Data should be a numpy array"
        assert len(
            self.data[0]) > 5, "Data should have at least 6 columns with same order with MOT17 format"
