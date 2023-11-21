import numpy as np
import cv2
import torch

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


sam_checkpoint = "third_party/clipasso/models/sam_vit_h_4b8939.pth"
model_type = "vit_h"
SAM = sam_model_registry[model_type](checkpoint=sam_checkpoint)


def check_pt_in_inds(pt, inds):
    for ind in inds:
        if pt[0] == ind[0] and pt[1] == ind[1]:
            return True
    return False


def get_outlier_indices(target_im, inds):
    outlier_indices = []
    distance_threshold = 16.0
    radius = 4
    boundary_min = 0
    boundary_max = target_im.shape[-1]
    for i in range(len(inds)):
        x, y = inds[i]

        distance_to_boundary = min(abs(x - boundary_min), abs(x - boundary_max), abs(y - boundary_min), abs(y - boundary_max))

        if distance_to_boundary <= distance_threshold:
            outlier_indices.append(i)
            continue

        x_min, x_max = max(0, x - radius), min(boundary_max, x + radius + 1)
        y_min, y_max = max(0, y - radius), min(boundary_max, y + radius + 1)

        t_image = target_im.squeeze().cpu().numpy()
        region = t_image[:, x_min:x_max, y_min:y_max]
        if np.all(region >= 0.99):
            outlier_indices.append(i)
    # outlier_indices = outlier_indices[::-1]
    return outlier_indices


def add_init_points_by_sam(input_im, device, pts_each_contour, thresh, inds):
    SAM.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(SAM)

    image = input_im.squeeze().permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()
    masks = mask_generator.generate(image)
    binary_mask = masks[0]['segmentation']
    binary_mask = binary_mask.astype(np.uint8) * 255

    pts_to_be_added_grouped, pts_to_be_added_sorted = [], []
    for i in range(len(masks)):
        binary_mask = masks[i]['segmentation']
        binary_mask = binary_mask.astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_points = contours[0]
        contour_points_scores = {}
        for cp in contour_points:
            x, y = cp[0][1], cp[0][0]
            contour_points_scores[(x, y)] = thresh[x][y]
        points_with_max_scores = [(k, v) for k, v in sorted(contour_points_scores.items(), key=lambda item: item[1], reverse=True)[:pts_each_contour]]

        final_points_with_max_scores = []
        for pt, sco in points_with_max_scores:
            x, y = pt
            if check_pt_in_inds(pt, inds):
                continue
            final_points_with_max_scores.append(([x, y], sco))

        pts_to_be_added_grouped.append([item[0] for item in final_points_with_max_scores])
        pts_to_be_added_sorted.extend(final_points_with_max_scores)

    pts_to_be_added_sorted = [item[0] for item in sorted(pts_to_be_added_sorted, key=lambda x: x[1], reverse=True)]
    return pts_to_be_added_grouped, pts_to_be_added_sorted


def random_point_around(xy, radius=2):
    theta = 2 * np.pi * np.random.rand()
    r = radius * np.sqrt(np.random.rand())
    x = xy[0] + r * np.cos(theta)
    y = xy[1] + r * np.sin(theta)
    return np.array([x, y])


if __name__ == "__main__":
    pass

