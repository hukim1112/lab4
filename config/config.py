class config():
	def __init__(self):
		self.input_shape = (256, 192)
		self.num_kps = 17
		self.rotation_factor = 40
		self.scale_factor = 0.3
		self.kps_symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
