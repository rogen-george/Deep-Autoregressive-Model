from PIL import Image
import glob
import pickle
import numpy as np
import cv2

pickle_file = open('spacinv.pkl','wb')

for f in glob.glob("Atari/*.png"):
    img2 = cv2.imread(f)
    if img2 is not None:
        img = np.asarray(Image.open(f))
        pickle.dump(img, pickle_file)

pickle_file.close()
