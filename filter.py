import cv2
import numpy as np


def filter_target_edge(
    canny_edges,
    side,
    min_area=20,
    min_length=10,
    return_info=False
):
    """
    根据左右区域信息，从 Canny 边缘中筛选目标边。

    参数:
        canny_edges:
            Canny 提取后的二值边缘图，单通道图像。
            边缘像素为 255，背景为 0。

        side:
            当前区域属于左侧还是右侧。
            可选：
                "left"  表示左侧目标区域
                "right" 表示右侧目标区域

        min_area:
            最小连通域面积，小于该面积的边会被认为是噪声边。

        min_length:
            最小边长度。
            对水平边，主要看连通域宽度；
            对竖直边，主要看连通域高度；
            这里使用 max(width, height) 作为边长度。

        return_info:
            是否返回被选中边的信息。

    返回:
        如果 return_info=False:
            filtered_edges

        如果 return_info=True:
            filtered_edges, selected_info

        filtered_edges:
            筛选后的边缘图，只保留目标边。

        selected_info:
            被选中边的信息，包括 bbox、center、area 等。
    """

    if canny_edges is None:
        if return_info:
            return None, None
        return None

    if len(canny_edges.shape) != 2:
        raise ValueError("canny_edges 必须是单通道二值图像")

    if side not in ["left", "right"]:
        raise ValueError("side 只能是 'left' 或 'right'")

    h, w = canny_edges.shape[:2]

    roi_center_x = w / 2.0
    roi_center_y = h / 2.0

    # 确保输入是 0/255 二值图
    edge_binary = np.zeros_like(canny_edges, dtype=np.uint8)
    edge_binary[canny_edges > 0] = 255

    # 找连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        edge_binary,
        connectivity=8
    )

    edge_candidates = []

    # label_id 从 1 开始，0 是背景
    for label_id in range(1, num_labels):
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        bw = stats[label_id, cv2.CC_STAT_WIDTH]
        bh = stats[label_id, cv2.CC_STAT_HEIGHT]
        area = stats[label_id, cv2.CC_STAT_AREA]

        cx, cy = centroids[label_id]

        edge_length = max(bw, bh)

        # 过滤过小的边
        if area < min_area:
            continue

        if edge_length < min_length:
            continue

        edge_candidates.append({
            "label_id": label_id,
            "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
            "center": [float(cx), float(cy)],
            "area": int(area),
            "width": int(bw),
            "height": int(bh),
            "length": int(edge_length)
        })

    # 如果没有候选边，返回空图
    filtered_edges = np.zeros_like(edge_binary, dtype=np.uint8)

    if len(edge_candidates) == 0:
        if return_info:
            return filtered_edges, None
        return filtered_edges

    # 如果只有一条边，直接保留
    if len(edge_candidates) == 1:
        selected = edge_candidates[0]
        filtered_edges[labels == selected["label_id"]] = 255

        if return_info:
            return filtered_edges, selected
        return filtered_edges

    # 如果有多条边，根据左右区域规则筛选
    valid_side_edges = []

    for item in edge_candidates:
        cx, cy = item["center"]

        if side == "left":
            # 左侧区域：选择 ROI 中心点右侧的边
            if cx >= roi_center_x:
                item["distance_to_center"] = abs(cx - roi_center_x)
                valid_side_edges.append(item)

        elif side == "right":
            # 右侧区域：选择 ROI 中心点左侧的边
            if cx <= roi_center_x:
                item["distance_to_center"] = abs(cx - roi_center_x)
                valid_side_edges.append(item)

    # 如果严格按左右规则没有找到边，则退化为选择距离中心最近的边
    if len(valid_side_edges) == 0:
        for item in edge_candidates:
            cx, cy = item["center"]
            item["distance_to_center"] = abs(cx - roi_center_x)

        valid_side_edges = edge_candidates

    # 选择距离区域中心点最近的那条边
    selected = min(
        valid_side_edges,
        key=lambda item: item["distance_to_center"]
    )

    # 生成最终筛选后的边缘图
    filtered_edges[labels == selected["label_id"]] = 255

    if return_info:
        return filtered_edges, selected

    return filtered_edges
