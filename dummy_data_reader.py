import numpy as np
from datetime import datetime, timezone

class DasFile:
    def __init__(self, file=None, datatype=1, nodata=False, nowarn=False):
        # Mocking the essential attributes extracted from the binary header
        self.frameLength = 2048
        self.prf = 2000
        self.rangeMeters = 10000.0
        self.resolution = 5.0
        self.sampleInterval = 5.0
        self.creationTime = datetime.now(timezone.utc)
        self.sampleType = 'RealUInt16'
        self.polarization = 1
        self.port = 1
        self.numFrames = 1000
        self.duration = self.numFrames / self.prf
        self.isComplex = False
        self.datatype = datatype

    def getData(self, frame_offset, frame_size):
        # Prevents out-of-bounds requests just like the original code
        if frame_offset > self.numFrames:
            return np.zeros((self.port, self.polarization, self.frameLength, 0))
        if frame_offset + frame_size > self.numFrames:
            frame_size = self.numFrames - frame_offset

        # Generate synthetic random noise to mimic acoustic data
        dummy_data = np.random.normal(loc=0.0, scale=1.0, size=(frame_size, self.frameLength, self.polarization, self.port))
        return dummy_data