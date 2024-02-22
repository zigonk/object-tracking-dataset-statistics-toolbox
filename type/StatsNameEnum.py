from enum import Enum


class StatsName(Enum):
    NUM_OBJ_PER_VIDEO = '#objects per video'
    NUM_OBJ_PER_FRAME = '#objects per frame'
    VIDEO_LENGTH = 'Video length'
    TRACK_GAP_LENGTH = 'Track gap length'
    IOU_RATIO_OBJECTS_INTRA_FRAME = 'IoU intra-frame'
    IOU_RATIO_TRACK_INTER_FRAME = 'IoU inter-frame'
