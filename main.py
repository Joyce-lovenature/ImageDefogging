import cv2 as cv
import numpy as np


class Defog:
	def __init__(self, path, window, a_thresh, omega, t_thresh):
		self.im = cv.imread(path)
		self.im = self.im.astype('float64')/255
		self.M, self.N, _ = self.im.shape
		self.window = window
		self.a_thresh = a_thresh
		self.omega = omega
		self.t_thresh = t_thresh

	def __get_dark_channel(self, im):
		w = self.window//2
		im_padding = np.pad(im, [[w, w], [w, w], [0, 0]], mode='reflect')
		im_dark = np.zeros([self.M, self.N])
		for i, j in np.ndindex((self.M, self.N)):
			im_dark[i, j] = np.min(im_padding[i:i+w, j:j+w, :])

		return im_dark

	def __get_atmosphere(self, im_dark):
		im_dark_flat = im_dark.flatten()
		im_flat = np.reshape(self.im, [self.M*self.N, 3])
		# im_gray_flat = cv.cvtColor(self.im, cv.COLOR_BGR2GRAY).flatten()
		t = self.M * self.N // 1000
		indexs = im_dark_flat.argsort()[-t:]
		# index = np.argmax(im_gray_flat.take(indexs))
		# return im_flat[index]
		a = np.mean(im_flat.take(indexs, axis=0), axis=0)
		a = np.minimum(a, self.a_thresh/255)
		return a

	def __get_t(self, a):
		return 1 - self.omega * self.__get_dark_channel(self.im/a)

	def __recovery(self, a, t):
		t_f = np.maximum(t, self.t_thresh)
		t_f = np.reshape(np.repeat(t_f, 3), [self.M, self.N, 3])
		im = np.float64(self.im)
		return (im - a)/t_f + a

	def defog_raw(self):
		dark = self.__get_dark_channel(engine.im)
		A = self.__get_atmosphere(dark)
		t = self.__get_t(A)
		i_r = self.__recovery(A, t)

		print(A)
		cv.imshow("t", t)
		cv.imshow("i_r", i_r)
		cv.waitKey(0)
		cv.destroyAllWindows()


if __name__ == "__main__":
	engine = Defog("images/city_fog.png", window=30, a_thresh=230, omega=0.95, t_thresh=0.1)
	engine.defog_raw()



