import cv2
import numpy as np

class MOB_Point_Cloud_Helper():
    def __init__(self):
        self.__h_min, self.__h_max = -180, 180
        self.__v_min, self.__v_max = -24.9, 2.0
        self.__v_res, self.__h_res = 0.42, 0.35
        self.__x, self.__y, self.__z, self.__d = None, None, None, None
        self.__h_fov, self.__v_fov = (-90, 90), (-24.9, 10.0)
        self.__x_range, self.__y_range, self.__z_range = None, None, None
        self.__get_sur_size, self.__get_top_size = None, None

    def __upload_points(self, points):
        self.__x = points[:, 0]
        self.__y = points[:, 1]
        self.__z = points[:, 2]
        self.__d = np.sqrt(self.__x ** 2 + self.__y ** 2 + self.__z ** 2)

    def __3d_in_range(self, points):
        """ extract filtered in-range velodyne coordinates based on x,y,z limit """
        return points[np.logical_and.reduce((self.__x > self.__x_range[0], self.__x < self.__x_range[1], \
                                             self.__y > self.__y_range[0], self.__y < self.__y_range[1], \
                                             self.__z > self.__z_range[0], self.__z < self.__z_range[1]))]
    def __points_filter(self, points):
        """
        filter points based on h,v FOV and x,y,z distance range.
        x,y,z direction is based on velodyne coordinates
        1. azimuth & elevation angle limit check
        2. x,y,z distance limit
        """

        # upload current points
        self.__upload_points(points)

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        if self.__h_fov is not None and self.__v_fov is not None:
            if self.__h_fov[1] == self.__h_max and self.__h_fov[0] == self.__h_min and \
                    self.__v_fov[1] == self.__v_max and self.__v_fov[0] == self.__v_min:
                pass
            elif self.__h_fov[1] == self.__h_max and self.__h_fov[0] == self.__h_min:
                con = self.__hv_in_range(d, z, self.__v_fov, fov_type='v')
                lim_x, lim_y, lim_z, lim_d = self.__x[con], self.__y[con], self.__z[con], self.__d[con]
                self.__x, self.__y, self.__z, self.__d = lim_x, lim_y, lim_z, lim_d
            elif self.__v_fov[1] == self.__v_max and self.__v_fov[0] == self.__v_min:
                con = self.__hv_in_range(x, y, self.__h_fov, fov_type='h')
                lim_x, lim_y, lim_z, lim_d = self.__x[con], self.__y[con], self.__z[con], self.__d[con]
                self.__x, self.__y, self.__z, self.__d = lim_x, lim_y, lim_z, lim_d
            else:
                h_points = self.__hv_in_range(x, y, self.__h_fov, fov_type='h')
                v_points = self.__hv_in_range(d, z, self.__v_fov, fov_type='v')
                con = np.logical_and(h_points, v_points)
                lim_x, lim_y, lim_z, lim_d = self.__x[con], self.__y[con], self.__z[con], self.__d[con]
                self.__x, self.__y, self.__z, self.__d = lim_x, lim_y, lim_z, lim_d
        else:
            pass

        if self.__x_range is None and self.__y_range is None and self.__z_range is None:
            pass
        elif self.__x_range is not None and self.__y_range is not None and self.__z_range is not None:
            # extract in-range points
            temp_x, temp_y = self.__3d_in_range(self.__x), self.__3d_in_range(self.__y)
            temp_z, temp_d = self.__3d_in_range(self.__z), self.__3d_in_range(self.__d)
            self.__x, self.__y, self.__z, self.__d = temp_x, temp_y, temp_z, temp_d
        else:
            raise ValueError("Please input x,y,z's min, max range(m) based on velodyne coordinates. ")

    def __normalize_data(self, val, min, max, scale, depth=False, clip=False):
        """ Return normalized data """
        if clip:
            # limit the values in an array
            np.clip(val, min, max, out=val)
        if depth:
            return (((max - val) / (max - min)) * scale).astype(np.uint8)
        else:
            return (((val - min) / (max - min)) * scale).astype(np.uint8)

    def __hv_in_range(self, m, n, fov, fov_type='h'):
        if fov_type == 'h':
            return np.logical_and(np.arctan2(n, m) > (-fov[1] * np.pi / 180), \
                                  np.arctan2(n, m) < (-fov[0] * np.pi / 180))
        elif fov_type == 'v':
            return np.logical_and(np.arctan2(n, m) < (fov[1] * np.pi / 180), \
                                  np.arctan2(n, m) > (fov[0] * np.pi / 180))
        else:
            raise NameError("fov type must be set between 'h' and 'v' ")

    def crop_pc_and_reformat(self, points):
        """ extract points corresponding to FOV setting """

        self.__points_filter(points)

        # Stack points in sequence horizontally
        xyz_v = np.hstack((self.__x[:, None], self.__y[:, None], self.__z[:, None]))
        xyz_v = xyz_v.T

        # stack (1,n) arrays filled with the number 1
        one_mat = np.full((1, xyz_v.shape[1]), 1)
        xyz_v = np.concatenate((xyz_v, one_mat), axis=0)

        # need dist info for points colour
        colour = self.__normalize_data(self.__d, min=1, max=70, scale=120, clip=True)

        return xyz_v, colour

    def add_projection_on_img(self, points, color, image):
        """ project converted velodyne points into camera image """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        for i in range(points.shape[1]):
            cv2.circle(hsv_image, (int(points[0][i]), int(points[1][i])), 2, (int(color[i]), 255, 255), -1)

        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    def add_bbox_on_img(self, image, bbox):
        """ add bbox on image and return bbox list"""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        img_width = image.shape[1]
        img_height = image.shape[0]
        bbox_list = []
        for i in range(bbox.shape[0]):
            x1, x2, y1, y2 = self.__convert_yolo2bbx(img_width, img_height, bbox[i, 1], bbox[i, 2], bbox[i, 3], bbox[i, 4])
            bbox_list.append([x1,y1,x2,y2])
            hsv_image = cv2.rectangle(hsv_image, (x1, y1), (x2, y2), (0, 255, 255), thickness=2)
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB), np.array(bbox_list)

    def read_lidar(self, lidar_path):
        """ read velodyne points from binary file """
        scan = np.fromfile(lidar_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan

    def read_image(self, image_path):
        """ read image from image file """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def read_label(self, label_path):
        yolo_labels = np.loadtxt(label_path)
        boxes = np.reshape(yolo_labels, (-1, 5))
        return boxes

    def __convert_yolo2bbx(self, width, height, x_center, y_center, bbox_width, bbox_height):
        x_c = x_center * width
        y_c = y_center * height
        bbox_w = bbox_width * width
        bbox_h = bbox_height * height
        x1 = x_c - bbox_w / 2
        x2 = x_c + bbox_w / 2
        y1 = y_c - bbox_h / 2
        y2 = y_c + bbox_h / 2
        return int(x1), int(x2), int(y1), int(y2)
