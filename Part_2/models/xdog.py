from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter
import numpy as np


class XDoG(object):
    def __init__(self):
        super(XDoG, self).__init__()
        self.gamma=0.98
        self.phi=200
        self.eps=-0.1
        self.sigma=0.8
        self.binarize=True
        
    def __call__(self, imgs, k=10):
        imdiffs = []
        for im in imgs:

            if im.shape[-1] == 3:
                im = rgb2gray(im)
            else:
                im = im.squeeze(2)

            imf1 = gaussian_filter(im, self.sigma)
            imf2 = gaussian_filter(im, self.sigma * k)
            imdiff = imf1 - self.gamma * imf2
            imdiff = (imdiff < self.eps) * 1.0  + (imdiff >= self.eps) * (1.0 + np.tanh(self.phi * imdiff))
            imdiff -= imdiff.min()
            imdiff /= imdiff.max() + 1e-8
            if self.binarize:
                th = threshold_otsu(imdiff)
                imdiff = imdiff >= th
            imdiff = imdiff.astype('float32')
            
            imdiffs.append(imdiff)

        return np.stack(imdiffs, 0)