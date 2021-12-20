import numpy as np
import cv2 as cv
from scipy.sparse import csc_matrix


class SoftMatting:
	def __init__(self, im, t, epsilon, lamb, window=1):
		self.im = im
		self.t = t
		self.window = window
		self.epsilon = epsilon
		self.lamb = lamb
		self.W, self.H, _ = im.shape
		self.N = self.W * self.H

	def get_laplacian(self):
		L = csc_matrix((self.N, self.N))
		window_size = self.window*2 + 1
		w_k = window_size**2
		for i in range(self.W - 2*self.window):
			if i % 20 == 0:
				print(i)
			for j in range(self.H - 2*self.window):
				im_window = self.im[i:i+window_size, j:j+window_size, :]
				t = np.array([im_window[:, :, 0].reshape(9),
							  im_window[:, :, 1].reshape(9),
							  im_window[:, :, 2].reshape(9)])
				e_k = np.cov(t)
				miu_k = np.mean(t, axis=1)
				u3 = np.eye(3)
				part_2 = np.linalg.inv(e_k + self.epsilon/w_k*u3)
				for m in range(w_k):
					for n in range(m, w_k):
						(m_i, m_j), (n_i, n_j), l_m, l_n = self.__find_position(m, n, i, j)
						part_1 = (self.im[m_i, m_j, :] - miu_k).T
						part_3 = (self.im[n_i, n_j, :] - miu_k)
						temp = -(1 + np.dot(np.dot(part_1, part_2), part_3))/w_k
						if m == n:
							temp += 1
						L[l_m, l_n] += temp

		return L

	def __find_position(self, m, n, i, j):
		m_i_off = m // 3
		m_j_off = m % 3
		n_i_off = n // 3
		n_j_off = n % 3
		m_i, m_j = m_i_off + i, m_j_off + j
		n_i, n_j = n_i_off + i, n_j_off + j
		l_m = m_j * self.W + m_i
		l_n = n_j * self.W + n_i
		return (m_i, m_j), (n_i, n_j), l_m, l_n

	def get_t(self):
		pass



