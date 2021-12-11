import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import interpolate

from Track import Track

track = Track.trackFromDataFile('lagunaSeca.json')

track.plotTrack()
plt.show()
plt.show()
plt.show()
plt.show()