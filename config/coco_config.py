class config():
    def __init__(self):
        self.image_path = '/home/kerry/prj/pose_repo/lab4/datasets/COCO/images'
        self.input_shape = (256, 192)
        self.num_kps = 17
        self.rotation_factor = 40
        self.scale_factor = 0.3
        self.kps_symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
        self.output_shape = (self.input_shape[0]//4, self.input_shape[1]//4)
        if self.output_shape[0] == 64:
            self.sigma = 2
        elif self.output_shape[0] == 96:
            self.sigma = 3
        self.pixel_means = [[[123.68, 116.78, 103.94]]]