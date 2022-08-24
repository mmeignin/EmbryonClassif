import imageio
import numpy as np

batchsize = 64
def batch(iterable, n=batchsize):
	l = len(iterable)
	for ndx in range(0, l, n):
		yield iterable[ndx:min(ndx + n, l)]

def get_number_frames(img):
	n = 0
	for i in range(10000):
		try:
			img.seek(i)
			n += 1
		except EOFError:
		# Not enough frames in img
			break
	return n

def sub_sample_frames(img):
	n = get_number_frames(img)
	idxes = list(range(n))

	list_frames = []
	for idx in idxes:
		img.seek(idx)
		img_frame = np.asarray(img)
		list_frames.append(img_frame)

	return list_frames
0
