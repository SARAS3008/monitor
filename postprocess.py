import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO


# =========================
# 1. 你的直线检测函数
# =========================

def extract_line_value(line_roi):
    """
    这里替换成你自己的直线提取函数。

    参数:
        line_roi: YOLO检测框中心80%的图像区域

    返回:
        value: 一个数值，比如直线度、毛刺高度、宽度等
    """

    # =========================
    # 在这里写你的真实检测逻辑
    # =========================

    value = 0.0

    return value


# =========================
# 2. 左右目标筛选 + 中心80% ROI裁剪
# =========================

def select_left_right_targets_with_inner_roi(
    image,
    yolo_result,
    target_class_id=None,
    conf_thres=0.20,
    inner_ratio=0.8
):
    """
    根据YOLO检测结果：
    1. 根据目标框中心点cx判断左侧/右侧目标
    2. 每侧只保留置信度最高的检测框
    3. 对最终框裁剪中心80%区域，作为直线检测ROI
    """

    h, w = image.shape[:2]
    image_center_x = w / 2.0

    left_best = None
    right_best = None

    boxes = yolo_result.boxes

    if boxes is None or len(boxes) == 0:
        return {
            "left": None,
            "right": None
        }

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(xyxy, confs, clss):
        if conf < conf_thres:
            continue

        if target_class_id is not None and cls_id != target_class_id:
            continue

        x1, y1, x2, y2 = box

        if x2 <= x1 or y2 <= y1:
            continue

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        det = {
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "conf": float(conf),
            "cls": int(cls_id),
            "center": [float(cx), float(cy)]
        }

        if cx < image_center_x:
            if left_best is None or conf > left_best["conf"]:
                left_best = det
        else:
            if right_best is None or conf > right_best["conf"]:
                right_best = det

    def add_roi(det):
        if det is None:
            return None

        x1, y1, x2, y2 = det["box"]

        # 原始检测框整数坐标
        x1_i = int(max(0, np.floor(x1)))
        y1_i = int(max(0, np.floor(y1)))
        x2_i = int(min(w, np.ceil(x2)))
        y2_i = int(min(h, np.ceil(y2)))

        if x2_i <= x1_i or y2_i <= y1_i:
            return None

        det["box_int"] = [x1_i, y1_i, x2_i, y2_i]
        det["target_roi"] = image[y1_i:y2_i, x1_i:x2_i].copy()

        # 计算中心80%区域
        box_w = x2 - x1
        box_h = y2 - y1

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        inner_w = box_w * inner_ratio
        inner_h = box_h * inner_ratio

        inner_x1 = cx - inner_w / 2.0
        inner_y1 = cy - inner_h / 2.0
        inner_x2 = cx + inner_w / 2.0
        inner_y2 = cy + inner_h / 2.0

        inner_x1_i = int(max(0, np.floor(inner_x1)))
        inner_y1_i = int(max(0, np.floor(inner_y1)))
        inner_x2_i = int(min(w, np.ceil(inner_x2)))
        inner_y2_i = int(min(h, np.ceil(inner_y2)))

        if inner_x2_i <= inner_x1_i or inner_y2_i <= inner_y1_i:
            return None

        det["inner_box"] = [
            inner_x1_i,
            inner_y1_i,
            inner_x2_i,
            inner_y2_i
        ]

        det["line_roi"] = image[
            inner_y1_i:inner_y2_i,
            inner_x1_i:inner_x2_i
        ].copy()

        return det

    left_best = add_roi(left_best)
    right_best = add_roi(right_best)

    return {
        "left": left_best,
        "right": right_best
    }


# =========================
# 3. 绘制单个检测结果
# =========================

def draw_one_result(image, det, side_name, value):
    """
    在原图上绘制：
    1. YOLO最终保留框
    2. 中心80%直线检测区域
    3. 检测数值
    """

    x1, y1, x2, y2 = det["box_int"]
    ix1, iy1, ix2, iy2 = det["inner_box"]

    conf = det["conf"]

    # YOLO最终保留框：绿色
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # 中心80%直线检测区域：红色
    cv2.rectangle(
        image,
        (ix1, iy1),
        (ix2, iy2),
        (0, 0, 255),
        2
    )

    # 数值文本
    if value is None:
        text = f"{side_name}: fail, conf={conf:.3f}"
    else:
        try:
            text = f"{side_name}: {float(value):.4f}, conf={conf:.3f}"
        except Exception:
            text = f"{side_name}: {value}, conf={conf:.3f}"

    text_x = x1
    text_y = max(30, y1 - 10)

    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


# =========================
# 4. 处理单张图片
# =========================

def process_one_image(
    image,
    model,
    extract_line_value_func,
    target_class_id=0,
    yolo_conf=0.15,
    conf_thres=0.20,
    inner_ratio=0.8
):
    """
    处理单张图片。

    返回:
        left_value: 左侧检测数值
        right_value: 右侧检测数值
        result_image: 画好结果的图像
    """

    result_image = image.copy()

    # YOLO检测
    results = model(image, conf=yolo_conf)
    yolo_result = results[0]

    # 左右目标筛选 + 中心80%区域裁剪
    selected = select_left_right_targets_with_inner_roi(
        image=image,
        yolo_result=yolo_result,
        target_class_id=target_class_id,
        conf_thres=conf_thres,
        inner_ratio=inner_ratio
    )

    left_value = None
    right_value = None

    # 左侧目标
    if selected["left"] is not None:
        left_line_roi = selected["left"]["line_roi"]

        try:
            left_value = extract_line_value_func(left_line_roi)
        except Exception as e:
            print("左侧直线检测失败:", e)
            left_value = None

        draw_one_result(
            result_image,
            selected["left"],
            side_name="left",
            value=left_value
        )
    else:
        cv2.putText(
            result_image,
            "left: not found",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

    # 右侧目标
    if selected["right"] is not None:
        right_line_roi = selected["right"]["line_roi"]

        try:
            right_value = extract_line_value_func(right_line_roi)
        except Exception as e:
            print("右侧直线检测失败:", e)
            right_value = None

        draw_one_result(
            result_image,
            selected["right"],
            side_name="right",
            value=right_value
        )
    else:
        cv2.putText(
            result_image,
            "right: not found",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

    # 画图像中心线，方便确认左右划分
    h, w = image.shape[:2]
    cv2.line(
        result_image,
        (w // 2, 0),
        (w // 2, h),
        (255, 0, 0),
        2
    )

    return left_value, right_value, result_image


# =========================
# 5. CSV写入
# =========================

def write_csv_header(csv_path):
    """
    创建CSV并写入表头。
    """

    with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "图像名称",
            "左侧检测结果",
            "右侧检测结果"
        ])


def append_result_to_csv(csv_path, image_name, left_value, right_value):
    """
    追加单张图片检测结果。
    """

    with open(csv_path, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            image_name,
            "" if left_value is None else left_value,
            "" if right_value is None else right_value
        ])


# =========================
# 6. 递归处理文件夹
# =========================

def process_folder(
    input_dir,
    output_dir,
    model_path,
    extract_line_value_func,
    target_class_id=0,
    yolo_conf=0.15,
    conf_thres=0.20,
    inner_ratio=0.8
):
    """
    递归处理 input_dir 及其子文件夹下的所有图片。

    输出:
        output_dir/
            detect_result.csv
            result_images/
                原始相对路径结构下的结果图
    """

    image_extensions = (
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
    )

    os.makedirs(output_dir, exist_ok=True)

    result_image_dir = os.path.join(output_dir, "result_images")
    os.makedirs(result_image_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "detect_result.csv")
    write_csv_header(csv_path)

    # 加载YOLO模型
    model = YOLO(model_path)

    total_count = 0
    success_count = 0

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if not filename.lower().endswith(image_extensions):
                continue

            total_count += 1

            image_path = os.path.join(root, filename)

            # 相对路径，用于避免子文件夹中重名图片覆盖
            relative_path = os.path.relpath(image_path, input_dir)
            relative_dir = os.path.dirname(relative_path)

            # 输出结果图路径，保持原来的子文件夹结构
            save_sub_dir = os.path.join(result_image_dir, relative_dir)
            os.makedirs(save_sub_dir, exist_ok=True)

            name, ext = os.path.splitext(filename)
            save_image_name = name + "_result" + ext
            save_image_path = os.path.join(save_sub_dir, save_image_name)

            image = cv2.imread(image_path)

            if image is None:
                print("图像读取失败:", image_path)

                append_result_to_csv(
                    csv_path=csv_path,
                    image_name=relative_path,
                    left_value=None,
                    right_value=None
                )
                continue

            print("正在处理:", relative_path)

            try:
                left_value, right_value, result_image = process_one_image(
                    image=image,
                    model=model,
                    extract_line_value_func=extract_line_value_func,
                    target_class_id=target_class_id,
                    yolo_conf=yolo_conf,
                    conf_thres=conf_thres,
                    inner_ratio=inner_ratio
                )

                cv2.imwrite(save_image_path, result_image)

                append_result_to_csv(
                    csv_path=csv_path,
                    image_name=relative_path,
                    left_value=left_value,
                    right_value=right_value
                )

                success_count += 1

                print(
                    "完成:",
                    relative_path,
                    "左侧:",
                    left_value,
                    "右侧:",
                    right_value
                )

            except Exception as e:
                print("处理失败:", image_path)
                print("错误信息:", e)

                append_result_to_csv(
                    csv_path=csv_path,
                    image_name=relative_path,
                    left_value=None,
                    right_value=None
                )

    print("全部处理完成")
    print("图片总数:", total_count)
    print("成功处理:", success_count)
    print("结果图保存路径:", result_image_dir)
    print("CSV结果路径:", csv_path)

    return csv_path, result_image_dir


# =========================
# 7. 主函数入口
# =========================

if __name__ == "__main__":

    # 输入图片文件夹
    input_dir = r"D:\your_input_images"

    # 输出结果文件夹
    output_dir = r"D:\your_output_results"

    # YOLO模型路径
    model_path = r"D:\your_model\best.pt"

    # 开始批量处理
    process_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        model_path=model_path,
        extract_line_value_func=extract_line_value,
        target_class_id=0,      # 如果不需要类别过滤，可以改成 None
        yolo_conf=0.15,         # YOLO推理置信度，可以适当低一点
        conf_thres=0.20,        # 后处理置信度阈值
        inner_ratio=0.8         # 取检测框中心80%区域做直线检测
    )