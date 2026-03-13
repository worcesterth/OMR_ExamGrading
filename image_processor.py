import cv2
import numpy as np


def fix_orientation(image):
    return image


def remove_shadows_aggressive(image):
    planes = cv2.split(image)
    result_planes = []
    for plane in planes:
        dilated_img = cv2.dilate(plane, np.ones((30, 30), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(norm_img)
    return cv2.merge(result_planes)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts, target_w=None, target_h=None):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    if target_w is None:
        target_w = max(int(np.sqrt(((br[0]-bl[0])**2) + ((br[1]-bl[1])**2))), int(np.sqrt(((tr[0]-tl[0])**2) + ((tr[1]-tl[1])**2))))
        target_h = max(int(np.sqrt(((tr[0]-br[0])**2) + ((tr[1]-br[1])**2))), int(np.sqrt(((tl[0]-bl[0])**2) + ((tl[1]-bl[1])**2))))
    dst = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (target_w, target_h))


def process_single_scan_from_memory(image_array):
    fixed_img = fix_orientation(image_array)
    debug_edge_img = remove_shadows_aggressive(fixed_img)
    h, w = debug_edge_img.shape[:2]
    ratio = h / 800.0
    resized = cv2.resize(debug_edge_img, (int(w/ratio), 800))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 150)
    dilated = cv2.dilate(edged, np.ones((5, 5), np.uint8), iterations=2)
    cnts = sorted(cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea, reverse=True)[:5]

    paper_cnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > (800*resized.shape[1]*0.1):
            paper_cnt = approx; break

    paper_warped = fixed_img if paper_cnt is None else four_point_transform(fixed_img, paper_cnt.reshape(4, 2)*ratio)
    p_h, p_w = paper_warped.shape[:2]
    p_ratio = p_h/800.0
    p_small = cv2.resize(paper_warped, (int(p_w/p_ratio), 800))
    gray_small = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(cv2.cvtColor(p_small, cv2.COLOR_BGR2GRAY))
    p_thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(gray_small, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cnts, _ = cv2.findContours(p_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    markers = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04*peri, True)
        if len(approx) == 4 and 100 < cv2.contourArea(c) < 5000:
            M = cv2.moments(c)
            if M["m00"] != 0: markers.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))

    if len(markers) >= 4:
        midX, midY = p_small.shape[1]//2, p_small.shape[0]//2
        tl = min([m for m in markers if m[0] < midX and m[1] < midY], key=lambda p: p[0]**2+p[1]**2, default=None)
        tr = min([m for m in markers if m[0] > midX and m[1] < midY], key=lambda p: (p[0]-p_small.shape[1])**2+p[1]**2, default=None)
        bl = min([m for m in markers if m[0] < midX and m[1] > midY], key=lambda p: p[0]**2+(p[1]-p_small.shape[0])**2, default=None)
        br = min([m for m in markers if m[0] > midX and m[1] > midY], key=lambda p: (p[0]-p_small.shape[1])**2+(p[1]-p_small.shape[0])**2, default=None)
        if all([tl, tr, bl, br]):
            return four_point_transform(paper_warped, np.array([tl, tr, br, bl])*p_ratio, 484, 700)
    return paper_warped


def find_and_read_subject_code(image, debug=True):
    roi_y, roi_x, roi_w, roi_h = 62, 357, 24, 126
    target_roi = image[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]
    debug_roi_color = target_roi.copy()

    gray_roi = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_roi, (3, 3), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cols = 2
    rows = 10
    cell_w = roi_w / cols
    cell_h = roi_h / rows
    detected_digits = []

    for j in range(cols):
        max_fill_ratio = -1
        best_row = -1
        for r in range(rows):
            x1 = int(j * cell_w)
            y1 = int(r * cell_h)
            x2 = int((j + 1) * cell_w)
            y2 = int((r + 1) * cell_h)
            cell_img = thresh[y1:y2, x1:x2]
            total_pixels = cv2.countNonZero(cell_img)
            area = (x2 - x1) * (y2 - y1)
            fill_ratio = total_pixels / float(area) if area > 0 else 0
            cv2.rectangle(debug_roi_color, (x1, y1), (x2, y2), (0, 0, 255), 1)
            if fill_ratio > max_fill_ratio and fill_ratio > 0.4:
                max_fill_ratio = fill_ratio
                best_row = r
        if best_row != -1:
            detected_digits.append(best_row)
            cx = int((j * cell_w) + (cell_w / 2))
            cy = int((best_row * cell_h) + (cell_h / 2))
            cv2.circle(debug_roi_color, (cx, cy), int(cell_w / 3), (0, 255, 0), -1)
        else:
            detected_digits.append("?")

    subject_code = "".join(map(str, detected_digits))
    return subject_code


def read_student_id(image, debug=True):
    bx = 395
    by = 62
    w = 90
    h = 120

    roi = image[by:by+h, bx:bx+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cols = 7
    rows = 10
    cell_w = w / cols
    cell_h = h / rows

    detected_digits = []
    debug_roi = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for j in range(cols):
        best_r = -1
        max_ratio = -1

        for r in range(rows):
            sx = int(j * cell_w)
            ex = int((j + 1) * cell_w)
            sy = int(r * cell_h)
            ey = int((r + 1) * cell_h)

            mask = np.zeros(thresh.shape, dtype="uint8")
            cX = int((sx + ex) / 2)
            cY = int((sy + ey) / 2)
            radius = int(min(cell_w, cell_h) / 2.5)
            cv2.circle(mask, (cX, cY), radius, 255, -1)

            masked_input = cv2.bitwise_and(thresh, thresh, mask=mask)
            total_pixels = cv2.countNonZero(masked_input)
            circle_area = np.pi * (radius ** 2)
            fill_ratio = total_pixels / circle_area if circle_area > 0 else 0

            if fill_ratio > max_ratio and fill_ratio > 0.45:
                max_ratio = fill_ratio
                best_r = r

            cv2.rectangle(debug_roi, (sx, sy), (ex, ey), (0, 0, 150), 1)

        if best_r != -1:
            detected_digits.append(best_r)
            res_sy = int(best_r * cell_h)
            res_cY = int(res_sy + (cell_h / 2))
            cv2.circle(debug_roi, (cX, res_cY), radius, (0, 255, 0), -1)
        else:
            detected_digits.append("?")

    student_id = "".join(map(str, detected_digits))
    return student_id


def read_exam_set_fixed(image, debug=False):
    roi = image[320:334, 25:147]
    score1 = np.sum(255 - cv2.cvtColor(roi[:, 10:20], cv2.COLOR_BGR2GRAY))
    score2 = np.sum(255 - cv2.cvtColor(roi[:, 90:100], cv2.COLOR_BGR2GRAY))
    if score1 > score2 + 200: return "1"
    elif score2 > score1 + 200: return "2"
    return "1" if score1 > score2 else "2"


def read_choice_answers_final(image, debug=False):
    blocks = [
        (1, 5,   32,  372, 60, 110),
        (6, 10,  110, 372, 60, 110),
        (11, 15, 188, 372, 60, 110),
        (16, 20, 265, 372, 60, 110),
        (21, 25, 344, 372, 60, 110)
    ]
    results = {}; choices_map = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5'}
    for (start_q, end_q, bx, by, w, h) in blocks:
        roi = image[by:by+h, bx:bx+w]
        thresh = cv2.threshold(cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        cell_h, cell_w = h / 5, w / 5
        for r in range(5):
            scores = []
            for c in range(5):
                cell = cleaned[int(r * cell_h):int((r + 1) * cell_h), int(c * cell_w):int((c + 1) * cell_w)]
                mask = np.zeros(cell.shape, dtype="uint8")
                cv2.circle(mask, (cell.shape[1]//2, cell.shape[0]//2), int(min(cell.shape[0], cell.shape[1])/2.8), 255, -1)
                scores.append(cv2.countNonZero(cv2.bitwise_and(cell, cell, mask=mask)))
            sorted_scores = sorted(scores, reverse=True)
            if sorted_scores[0] < 7: results[start_q + r] = "Empty"
            elif sorted_scores[1] > (sorted_scores[0] * 0.7): results[start_q + r] = "Double"
            else: results[start_q + r] = choices_map[np.argmax(scores)]
    return results


def read_numeric_answers_advanced(image, debug=False):
    numeric_configs = {
        26: {'int': (13, 565, 44, 108), 'dec': (65, 565, 22, 108)},
        27: {'int': (108, 565, 44, 108), 'dec': (162, 565, 22, 108)},
        28: {'int': (200, 565, 44, 108), 'dec': (256, 565, 22, 108)},
        29: {'int': (297, 565, 44, 108), 'dec': (352, 565, 22, 108)},
        30: {'int': (393, 565, 44, 108), 'dec': (447, 565, 22, 108)}
    }
    final_results = {}
    for q_num, parts in numeric_configs.items():
        q_value_str = ""
        for part_type in ['int', 'dec']:
            if part_type not in parts: continue
            bx, by, w, h = parts[part_type]
            num_cols = 4 if part_type == 'int' else 2
            roi = image[by:by+h, bx:bx+w]
            thresh = cv2.threshold(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            cell_w, cell_h, part_res = w / num_cols, h / 10, ""
            for c in range(num_cols):
                ratios = []
                for r in range(10):
                    cell = cleaned[int(r * cell_h):int((r + 1) * cell_h), int(c * cell_w):int((c + 1) * cell_w)]
                    mask = np.zeros(cell.shape, dtype="uint8")
                    cv2.circle(mask, (cell.shape[1]//2, cell.shape[0]//2), int(min(cell.shape[0], cell.shape[1])/2.8), 255, -1)
                    ratios.append(cv2.countNonZero(cv2.bitwise_and(cell, cell, mask=mask)) / max(1, cv2.countNonZero(mask)))
                sorted_indices = np.argsort(ratios)[::-1]
                if ratios[sorted_indices[0]] < 0.15: part_res += "X"
                elif ratios[sorted_indices[1]] > (ratios[sorted_indices[0]] * 0.7): part_res += "D"
                else: part_res += str(sorted_indices[0])
            q_value_str += part_res + "." if part_type == 'int' else part_res
        final_results[q_num] = q_value_str
    return final_results


def read_choice_answers_50q_no_cross(image, debug=False):
    blocks = [
        (1, 10,   39,  407, 64, 250),
        (11, 20,  130, 407, 64, 250),
        (21, 30,  218, 407, 64, 250),
        (31, 40,  308, 407, 64, 250),
        (41, 50,  397, 407, 64, 250)
    ]
    choices_map = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5'}
    results = {}
    for (start_q, end_q, bx, by, w, h) in blocks:
        roi = image[by:by+h, bx:bx+w]
        thresh_raw = cv2.adaptiveThreshold(cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
        cleaned = cv2.morphologyEx(thresh_raw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        cell_h, cell_w = h / 10, w / 5
        for r in range(10):
            ratios = []
            for c in range(5):
                cell = cleaned[int(r * cell_h):int((r + 1) * cell_h), int(c * cell_w):int((c + 1) * cell_w)]
                mask = np.zeros(cell.shape, dtype="uint8")
                cv2.circle(mask, (mask.shape[1]//2, mask.shape[0]//2), int(min(cell_h, cell_w)/2.8), 255, -1)
                ratios.append(cv2.countNonZero(cv2.bitwise_and(cell, cell, mask=mask)) / max(1, cv2.countNonZero(mask)))
            sorted_indices = np.argsort(ratios)[::-1]
            if ratios[sorted_indices[0]] >= 0.1:
                results[start_q + r] = "Double" if ratios[sorted_indices[1]] > (ratios[sorted_indices[0]] * 0.8) else choices_map[sorted_indices[0]]
            else:
                results[start_q + r] = "Empty"
    return results
