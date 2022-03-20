import numpy as np

class mpii_skeleton:
    parents_data = [1, 2, 6, 6, 3, 4, -1, 6, 7, 8, 11, 12, 8, 8, 13, 14]

    def parents():
        return mpii_skeleton.parents_data

    def num_joints():
        return len(mpii_skeleton.parents_data)

class coco_reduce_skeleton:
    # Reduce: 0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee,
    #         5 - l ankle, 6 - head top,
    #         7 - r wrist, 8 - r elbow, 9 - r shoulder, 10 - l shoulder,
    #         11 - l elbow, 12 - l wrist
    parents_data = [1, 2, 9, 10, 3, 4, -1, 8, 9, 6, 6, 10, 11]

    def parents():
        return coco_reduce_skeleton.parents_data

    def num_joints():
        return len(coco_reduce_skeleton.parents_data)
