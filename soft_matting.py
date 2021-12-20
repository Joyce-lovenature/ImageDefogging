import numpy as np
import cv2 as cv
from scipy.sparse import csc_matrix


class SoftMatting:
	def __init__(self, im, t, epsilon, lamb, window=3):
		self.im = im
		self.t = t
		self.window = window
		self.epsilon = epsilon
		self.lamb = lamb
		self.W, self.H, _ = im.shape
		self.N = self.W * self.H
		print(self.N)
		self.part2_matrix = np.zeros([self.W, self.H, self.window, self.window])
		self.miu_matrix = np.zeros([self.W, self.H, self.window])

	def get_laplacian(self):
		L = csc_matrix((self.N, self.N))
		for i in range(self.N):
			if i % 500 == 0:
				print(i)
			for j in range(i, self.N):
				value = self.get_laplacian_elem(i, j)
				if value != 0:
					L[i, j] = L[j, i] = value
		return L

	def get_laplacian_elem(self, i, j):
		i_r = i // self.H
		i_c = i % self.H
		j_r = j // self.H
		j_c = j % self.H
		if abs(i_c - j_c) > self.window or abs(i_r - j_r) > self.window:
			return 0
		r_start = max(0, max(i_r, j_r) - self.window + 1)
		r_end = min(self.W - self.window, min(i_r, j_r) + 1)
		c_start = max(0, min(i_c, j_c) - self.window + 1)
		c_end = min(self.H - self.window, max(i_c, j_c) + 1)

		w_k = self.window**2
		value = 0

		for r in range(r_start, r_end):
			for c in range(c_start, c_end):
				im_window = self.im[r:r + self.window, c:c + self.window, :]
				t = np.array([im_window[:, :, 0].reshape(9),
							  im_window[:, :, 1].reshape(9),
							  im_window[:, :, 2].reshape(9)])
				if self.miu_matrix[r, c, 0] == 0:
					e_k = np.cov(t)
					miu_k = self.miu_matrix[r, c, :] = np.mean(t, axis=1)
					u3 = np.eye(3)
					part_2 = self.part2_matrix[r, c, :, :] = np.linalg.inv(e_k + self.epsilon / w_k * u3)
				else:
					miu_k = self.miu_matrix[r, c, :]
					part_2 = self.part2_matrix[r, c, :, :]
				part_1 = (self.im[i_r, i_c, :] - miu_k).T
				part_3 = self.im[j_r, j_c, :] - miu_k
				temp = -(1 + np.dot(np.dot(part_1, part_2), part_3))/w_k
				if i == j:
					temp += 1
				value += temp
		return value

	def get_t(self):
		pass



