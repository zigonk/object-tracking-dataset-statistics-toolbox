from enum import Enum

class StatsName(Enum):
    NUM_OBJ_PER_VIDEO = 'Number of objects per video'
    NUM_OBJ_PER_FRAME = 'Number of objects per frame'
    VIDEO_LENGTH = 'Video length'
    TRACK_GAP_LENGTH = 'Track gap length'
    IOU_RATIO_OBJECTS_INTRA_FRAME = 'IOU ratio between objects in the same frame'
    IOU_RATIO_TRACK_INTER_FRAME = 'IOU ratio between tracks in consecutive frames'
    
