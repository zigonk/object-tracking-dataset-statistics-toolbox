from typing import Any, List
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


class TrackingQuery(object):
    def __init__(self, data):
        # Example of data:
        # {
        #     "class_name": str,
        #     "synonyms": List[str],
        #     "type": str,
        #     "is_eval": bool,
        #     "definition": str,
        #     "attributes": List[str],
        #     "video_path": str,
        #     "track_path": str,
        #     "caption": str
        # },
        self.class_name: str = data['class_name']
        self.synonyms: List[str] = data['synonyms'] if 'synonyms' in data else [
        ]
        self.type: str = data['type']
        self.is_eval: bool = data['is_eval']
        self.definition: str = data['definition']
        self.attributes: List[str] = data['attributes']
        self.video_path: str = data['video_path']
        self.track_path: str = data['track_path']
        self.caption: str = data['caption']
