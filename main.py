import cv2 as cv
import numpy as np
from scipy.ndimage import minimum_filter
from soft_matting import SoftMatting


class Defog:

    def __init__(self, path, window, a_thresh, omega, t_thresh):
        self.im_ori = cv.imread(path)
        self.im = self.im_ori.astype('float64') / 255
        self.M, self.N, _ = self.im.shape
        self.window = window
        self.a_thresh = a_thresh
        self.omega = omega
        self.t_thresh = t_thresh

    def __get_dark_channel(self, im):
        # w = self.window//2
        # im_padding = np.pad(im, [[w, w], [w, w], [0, 0]], mode='reflect')
        # im_dark = np.zeros([self.M, self.N])
        # for i, j in np.ndindex((self.M, self.N)):
        # 	im_dark[i, j] = np.min(im_padding[i:i+w, j:j+w, :])\
        im_dark = np.min(im, axis=2)
        im_dark = minimum_filter(im_dark, [self.window, self.window])

        return im_dark

    def __get_atmosphere(self, im_dark):
        im_dark_flat = im_dark.flatten()
        im_flat = np.reshape(self.im, [self.M * self.N, 3])
        # im_gray_flat = cv.cvtColor(self.im, cv.COLOR_BGR2GRAY).flatten()
        t = self.M * self.N // 1000
        indexs = im_dark_flat.argsort()[-t:]
        # index = np.argmax(im_gray_flat.take(indexs))
        # return im_flat[index]
        a = np.mean(im_flat.take(indexs, axis=0), axis=0)
        a = np.minimum(a, self.a_thresh / 255)
        return a

    def __get_t(self, a):
        return 1 - self.omega * self.__get_dark_channel(self.im / a)

    def __recovery(self, a, t):
        t_f = np.maximum(t, self.t_thresh)
        t_f = np.reshape(np.repeat(t_f, 3), [self.M, self.N, 3])
        im = np.float64(self.im)
        return (im - a) / t_f + a

    def __guided_filter_t(self, t, r = 60, eps = 0.0001):
        # r: window size
        Gray_img = cv.cvtColor(self.im_ori, cv.COLOR_RGB2GRAY)
        Gray_img = np.float64(Gray_img)/255
        I = cv.boxFilter(Gray_img, -1, (r, r))
        P = cv.boxFilter(t, -1, (r, r))
        IP = cv.boxFilter(Gray_img * t, -1, (r, r))
        sigma_2 = cv.boxFilter(Gray_img*Gray_img, -1, (r, r)) - I*I
        a = (IP - I*P) / (sigma_2 + eps)
        b = P - a * I
        a_mean = cv.boxFilter(a, -1, (r, r))
        b_mean = cv.boxFilter(b, -1, (r, r))

        gf_img = a_mean * Gray_img + b_mean
        return gf_img

    def defog_raw(self):
        dark = self.__get_dark_channel(self.im)
        A = self.__get_atmosphere(dark)
        t = self.__get_t(A)
        i_t = self.__recovery(A, t)

        print(A)
        cv.imshow("t", t)
        cv.imshow("defog with t", i_t)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def defog_gf(self):
        dark = self.__get_dark_channel(self.im)
        A = self.__get_atmosphere(dark)
        t = self.__get_t(A)
        Guided_Filtered_T = self.__guided_filter_t(t)
        i_t = self.__recovery(A, t)
        i_gf = self.__recovery(A, Guided_Filtered_T)

        print(A)
        cv.imshow("t", t)
        cv.imshow("gf_t", Guided_Filtered_T)
        cv.imshow("defog with t", i_t)
        cv.imshow("defog with gf_t", i_gf)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def defog_soft_matting(self):
        dark = self.__get_dark_channel(self.im)
        A = self.__get_atmosphere(dark)
        t = self.__get_t(A)
        soft_matting = SoftMatting(self.im, t, epsilon=0.0001, lamb=0.0001)
        L = soft_matting.get_laplacian()
        return L


if __name__ == "__main__":
    engine = Defog("images/city_fog.png", window=15, a_thresh=230, omega=0.95, t_thresh=0.1)
    L = engine.defog_soft_matting()
