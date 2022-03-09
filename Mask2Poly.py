import numpy as np
import cv2
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon


def mask_to_poly(image_path, mask_path, cat_id, image_id):
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    retries, threshold = cv2.threshold(mask, 120, 255, 0)
    contours = measure.find_contours(threshold, 0.5, positive_orientation='low')
    seg, polys = [], []
    for cnt in contours:
        area = cv2.contourArea(np.expand_dims(cnt.astype(np.float32), 1))
        if area > 50:
            poly = Polygon(cnt).simplify(2, preserve_topology=False)
            polys.append(poly)
            segmentation = np.around(np.array(poly.exterior.coords).ravel().tolist(), 2).tolist()
            seg.append(segmentation)

    poly_combined = MultiPolygon(polys)
    x, y, max_x, max_y = poly_combined.bounds
    bbox = (int(x), int(y), int(max_x - x), int(max_y - y))
    area = poly_combined.area
    annotation = {
            "segmentation": seg,
            "iscrowd": 0,
            "image_id": image_id,
            "category_id": cat_id,
            "id": 0,
            "bbox": bbox,
            "area": np.around(area, 2)
    }
    #image = cv2.imread(image_path)
    #cv2.rectangle(image, pt1=(int(x), int(y)), pt2=(int(max_x), int(max_y)), color=(255, 0, 0), thickness=1)
    polygon_coords = []

    for poly_ in polys:
        pts = np.asarray(poly_.exterior.coords)
        for point in range(len(pts)):
            pts[point][0], pts[point][1] = pts[point][1], pts[point][0]
        #cv2.polylines(image, np.int32([pts]), True, (0, 255, 0), thickness=1)
        polygon_coords.append(pts)
    #cv2.imshow("Main", image)
    #cv2.waitKey(0)
    return annotation, polygon_coords
