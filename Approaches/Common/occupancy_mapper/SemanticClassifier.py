import numpy as np
from ....Common.Utils.carla_utils import get_depth_in_meters


# REVIEW: useless without a semantic_segmentation sensor
class SemanticClassifier:
    name = "class to classify semantic segmentation"

    def __init__(self):
        self.semantic_segmentation_dic = {
            "Unlabeled": (0, 0, 0),
            "Building": (70, 70, 70),
            "Fence": (190, 153, 153),
            "Other": (250, 170, 160),
            "Pedestrian": (220, 20, 60),
            "Pole": (153, 153, 153),
            "Road line": (157, 234, 50),
            "Road": (128, 64, 128),
            "Sidewalk": (244, 35, 232),
            "Vegetation": (107, 142, 35),
            "Car": (0, 0, 142),
            "Wall": (102, 102, 156),
            "Traffic sign": (220, 220, 0),
            "Occluded": (255, 255, 255),
            "Obstacle": (100, 100, 100),
            "Free": (200, 200, 200)
        }
        self.segmentation_classification = {
            "road": ["Road", "Road line"],
            "dynamic": ["Pedestrian", "Car"],
            "sign": ["Traffic sign"],
            "line": ["Road line"],
            "static": ["Sidewalk", "Unlabeled", "Building", "Fence", "Pole", "Vegetation", "Wall", "Other"],
            "occluded":  ["Occluded"],
            "obstacles": ["Pedestrian", "Car", "Sidewalk", "Unlabeled"],
            "free": ["Free"],
            "non-free": ["Obstacle"]
        }
        self.color_classification = self.get_color_classification()
        self.max_index_near_pixel = 10  # value to tune

    def get_color_classification(self):
        color_classification = {}
        for class_type in self.segmentation_classification:
            color = []
            for sem in self.segmentation_classification[class_type]:
                color.append(self.semantic_segmentation_dic[sem])
            color_classification[class_type] = color
        return color_classification

    # def get_segmented_rs(self, rs_semantic, semantic_name):
    #     is_safe = True
    #     rs_test = []
    #     for t, rs_slice in enumerate(rs_semantic):
    #         test_slice = []
    #         for i, semantic in enumerate(rs_slice):
    #             if (semantic == self.color_classification[semantic_name]).any():
    #                 test_slice.append(True)
    #                 is_safe = False
    #             else:
    #                 test_slice.append(False)
    #         rs_test += [test_slice]
    #     return rs_test, is_safe

    def get_segmented_image(self, image, semantic_name):
        out = []
        for i in range(len(self.color_classification[semantic_name])):
            out.append(np.all(np.equal(image, self.color_classification[semantic_name][i]), axis=2))
        for i in range(len(self.color_classification[semantic_name])-1):
            out[i+1] = np.logical_or(out[i], out[i+1])
        return np.array(out[-1], dtype=np.uint8)*255

    def get_segmented_pixel(self, image, pixel, semantic_name):
        return self.get_segmented_image([[image[int(pixel[1])][int(pixel[0])]]], semantic_name)[0][0]

    def pixel_is_semantic(self, pixel_semantic, semantic_name):
        return self.get_segmented_image([[pixel_semantic]], semantic_name)[0][0]

    def find_nearest_pixel(self, pixel, image, semantic_name, depth_image, rec):
        target_depth = pixel[2]
        pixel_semantic = image[int(round(pixel[1]))][int(round(pixel[0]))]
        is_semantic_name = self.pixel_is_semantic(pixel_semantic, semantic_name)
        i = 0
        if is_semantic_name:
            return pixel
        potential_pixels = None
        while not is_semantic_name:
            i += 1
            if i > self.max_index_near_pixel:
                return None
            potential_pixels = []
            # x = pix_x - i; y =  [ pix_y - i ... pix_y + i ]
            for j in range(-i, i + 1, 1):
                if (int(round(pixel[0])) + j < rec["semantic_segmentation"].height
                    or int(round(pixel[0])) + j >= 0) \
                        and (int(round(pixel[1])) - i < rec["semantic_segmentation"].width
                             or int(round(pixel[1])) - i >= 0):
                    near_pixel_semantic = image[int(round(pixel[1])) - i][int(round(pixel[0])) + j]
                    if self.pixel_is_semantic(near_pixel_semantic, semantic_name):
                        potential_pixels.append([int(round(pixel[0])) + j, int(round(pixel[1])) - i])
                # x = pix_x + i; y =  [ pix_y - i ... pix_y + i ]
                if (int(round(pixel[0])) + j < rec["semantic_segmentation"].height
                    or int(round(pixel[0])) + j >= 0) \
                        and (int(round(pixel[1])) + i < rec["semantic_segmentation"].width
                             or int(round(pixel[1])) + i >= 0):
                    near_pixel_semantic = image[int(round(pixel[1])) + i][int(round(pixel[0])) + j]
                    if self.pixel_is_semantic(near_pixel_semantic, semantic_name):
                        potential_pixels.append([int(round(pixel[0])) + j, int(round(pixel[1])) + i])
            # x = ] pix_x - i ... pix_x + i [; y =  pix_y - i
            for j in range(-i + 1, i, 1):
                if (int(round(pixel[0])) - i < rec["semantic_segmentation"].height
                    or int(round(pixel[0])) - i >= 0) \
                        and (int(round(pixel[1])) + j < rec["semantic_segmentation"].width
                             or int(round(pixel[1])) + j >= 0):
                    near_pixel_semantic = image[int(round(pixel[1])) + j][int(round(pixel[0])) - i]
                    if self.pixel_is_semantic(near_pixel_semantic, semantic_name):
                        potential_pixels.append([int(round(pixel[0])) - i, int(round(pixel[1])) + j])
                # x = ] pix_x - i ... pix_x + i [; y =  pix_y + i
                if (int(round(pixel[0])) + i < rec["semantic_segmentation"].height
                    or int(round(pixel[0])) + i >= 0) \
                        and (int(round(pixel[1])) + j < rec["semantic_segmentation"].width
                             or int(round(pixel[1])) + j >= 0):
                    near_pixel_semantic = image[int(round(pixel[1])) + j][int(round(pixel[0])) + i]
                    if self.pixel_is_semantic(near_pixel_semantic, semantic_name):
                        potential_pixels.append([int(round(pixel[0])) + i, int(round(pixel[1])) + j])
            if len(potential_pixels) != 0:
                is_semantic_name = True
        # All pixels in potential_pixel are almost as near to pixel as each others (for small i)
        # We get the nearest with depth
        dist = float('inf')
        near_pixel = None
        near_pixel_depth = None
        for potential_pixel in potential_pixels:
            depth = get_depth_in_meters(depth_image[int(round(potential_pixel[1]))][int(round(potential_pixel[0]))])
            if abs(target_depth - depth) < dist:
                near_pixel = potential_pixel
                near_pixel_depth = depth
                dist = abs(target_depth - depth)
        return np.array([near_pixel[0], near_pixel[1], near_pixel_depth])
