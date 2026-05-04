import os  # 导入 os 模块，用于文件夹遍历、路径拼接、创建文件夹等操作
import csv  # 导入 csv 模块，用于把检测结果写入 CSV 文件
import cv2  # 导入 OpenCV 模块，用于图像读取、保存、画框、写文字等操作
import numpy as np  # 导入 numpy 模块，用于数组计算和坐标取整等操作
from ultralytics import YOLO  # 从 ultralytics 中导入 YOLO 模型类，用于加载和执行目标检测模型
# =========================  # 分隔线，用于区分代码模块
# 1. 你的直线检测函数  # 当前模块用于放置你的直线提取算法
# =========================  # 分隔线，用于区分代码模块
def extract_line_value(line_roi):  # 定义直线检测函数，输入为中心 80% 的 ROI 图像
    """  # 函数说明文档开始
    这里替换成你自己的直线提取函数。  # 说明这里需要换成你的真实算法
    参数:  # 参数说明
        line_roi: YOLO检测框中心80%的图像区域  # line_roi 是后续直线检测使用的局部图像
    返回:  # 返回值说明
        value: 一个数值，比如直线度、毛刺高度、宽度等  # 返回值是你需要记录和显示的检测结果
    """  # 函数说明文档结束
    value = 0.0  # 示例返回值，实际使用时需要替换成你的直线检测结果
    return value  # 返回直线检测得到的数值
# =========================  # 分隔线，用于区分代码模块
# 2. 左右目标筛选 + 中心80% ROI裁剪  # 当前模块用于从 YOLO 检测结果中选出左右目标并裁剪检测区域
# =========================  # 分隔线，用于区分代码模块
def select_left_right_targets_with_inner_roi(image, yolo_result, target_class_id=None, conf_thres=0.20, inner_ratio=0.8):  # 定义左右目标筛选函数
    """  # 函数说明文档开始
    根据YOLO检测结果：  # 说明该函数处理 YOLO 的检测框结果
    1. 根据目标框中心点cx判断左侧/右侧目标  # 第一步，根据中心点横坐标判断左右
    2. 每侧只保留置信度最高的检测框  # 第二步，如果一侧有多个框，只保留最高置信度
    3. 对最终框裁剪中心80%区域，作为直线检测ROI  # 第三步，裁剪每个目标框内部 80% 区域
    """  # 函数说明文档结束
    h, w = image.shape[:2]  # 获取图像高度 h 和宽度 w
    image_center_x = w / 2.0  # 计算图像中心线的 x 坐标，用于区分左侧和右侧
    left_best = None  # 初始化左侧最佳目标为空
    right_best = None  # 初始化右侧最佳目标为空
    boxes = yolo_result.boxes  # 从 YOLO 单张图像结果中取出所有检测框
    if boxes is None or len(boxes) == 0:  # 判断是否没有任何检测框
        return {"left": None, "right": None}  # 如果没有检测结果，直接返回左右都为空
    xyxy = boxes.xyxy.cpu().numpy()  # 获取检测框坐标，格式为 x1,y1,x2,y2，并转成 numpy 数组
    confs = boxes.conf.cpu().numpy()  # 获取每个检测框的置信度，并转成 numpy 数组
    clss = boxes.cls.cpu().numpy().astype(int)  # 获取每个检测框的类别 ID，并转成整数数组
    for box, conf, cls_id in zip(xyxy, confs, clss):  # 遍历每一个检测框、置信度和类别
        if conf < conf_thres:  # 如果当前检测框置信度小于后处理阈值
            continue  # 跳过当前检测框
        if target_class_id is not None and cls_id != target_class_id:  # 如果指定了类别，并且当前框类别不等于指定类别
            continue  # 跳过当前检测框
        x1, y1, x2, y2 = box  # 解包检测框坐标
        if x2 <= x1 or y2 <= y1:  # 判断检测框宽度或高度是否无效
            continue  # 如果检测框无效，则跳过
        cx = (x1 + x2) / 2.0  # 计算检测框中心点 x 坐标
        cy = (y1 + y2) / 2.0  # 计算检测框中心点 y 坐标
        det = {"box": [float(x1), float(y1), float(x2), float(y2)], "conf": float(conf), "cls": int(cls_id), "center": [float(cx), float(cy)]}  # 构建当前检测框信息字典
        if cx < image_center_x:  # 如果当前检测框中心点在图像中心线左侧
            if left_best is None or conf > left_best["conf"]:  # 如果左侧还没有目标，或者当前框置信度更高
                left_best = det  # 更新左侧最佳检测框
        else:  # 如果当前检测框中心点在图像中心线右侧或正好等于中心线
            if right_best is None or conf > right_best["conf"]:  # 如果右侧还没有目标，或者当前框置信度更高
                right_best = det  # 更新右侧最佳检测框
    def add_roi(det):  # 定义内部函数，用于给最终检测框添加原框 ROI 和中心 80% ROI
        if det is None:  # 判断传入的检测框是否为空
            return None  # 如果为空，直接返回 None
        x1, y1, x2, y2 = det["box"]  # 取出检测框的浮点坐标
        x1_i = int(max(0, np.floor(x1)))  # 将左上角 x1 向下取整，并限制不能小于 0
        y1_i = int(max(0, np.floor(y1)))  # 将左上角 y1 向下取整，并限制不能小于 0
        x2_i = int(min(w, np.ceil(x2)))  # 将右下角 x2 向上取整，并限制不能超过图像宽度
        y2_i = int(min(h, np.ceil(y2)))  # 将右下角 y2 向上取整，并限制不能超过图像高度
        if x2_i <= x1_i or y2_i <= y1_i:  # 判断整数化后的检测框是否有效
            return None  # 如果检测框无效，则返回 None
        det["box_int"] = [x1_i, y1_i, x2_i, y2_i]  # 保存整数形式的原始检测框坐标
        det["target_roi"] = image[y1_i:y2_i, x1_i:x2_i].copy()  # 从原图中裁剪完整目标框区域并保存
        box_w = x2 - x1  # 计算原始检测框的宽度
        box_h = y2 - y1  # 计算原始检测框的高度
        cx = (x1 + x2) / 2.0  # 重新计算检测框中心点 x 坐标
        cy = (y1 + y2) / 2.0  # 重新计算检测框中心点 y 坐标
        inner_w = box_w * inner_ratio  # 计算内部 ROI 的宽度，默认为原框宽度的 80%
        inner_h = box_h * inner_ratio  # 计算内部 ROI 的高度，默认为原框高度的 80%
        inner_x1 = cx - inner_w / 2.0  # 计算内部 ROI 左上角 x 坐标
        inner_y1 = cy - inner_h / 2.0  # 计算内部 ROI 左上角 y 坐标
        inner_x2 = cx + inner_w / 2.0  # 计算内部 ROI 右下角 x 坐标
        inner_y2 = cy + inner_h / 2.0  # 计算内部 ROI 右下角 y 坐标
        inner_x1_i = int(max(0, np.floor(inner_x1)))  # 内部 ROI 左上角 x 向下取整并防止越界
        inner_y1_i = int(max(0, np.floor(inner_y1)))  # 内部 ROI 左上角 y 向下取整并防止越界
        inner_x2_i = int(min(w, np.ceil(inner_x2)))  # 内部 ROI 右下角 x 向上取整并防止越界
        inner_y2_i = int(min(h, np.ceil(inner_y2)))  # 内部 ROI 右下角 y 向上取整并防止越界
        if inner_x2_i <= inner_x1_i or inner_y2_i <= inner_y1_i:  # 判断内部 ROI 是否有效
            return None  # 如果内部 ROI 无效，则返回 None
        det["inner_box"] = [inner_x1_i, inner_y1_i, inner_x2_i, inner_y2_i]  # 保存内部 80% 区域在原图上的坐标
        det["line_roi"] = image[inner_y1_i:inner_y2_i, inner_x1_i:inner_x2_i].copy()  # 裁剪内部 80% 区域作为直线检测输入
        return det  # 返回添加了 ROI 信息的检测结果字典
    left_best = add_roi(left_best)  # 给左侧最佳检测框添加 ROI 信息
    right_best = add_roi(right_best)  # 给右侧最佳检测框添加 ROI 信息
    return {"left": left_best, "right": right_best}  # 返回左右目标的最终结果
# =========================  # 分隔线，用于区分代码模块
# 3. 绘制单个检测结果  # 当前模块用于在图像上画检测框、内部框和检测数值
# =========================  # 分隔线，用于区分代码模块
def draw_one_result(image, det, side_name, value):  # 定义单个目标绘制函数
    """  # 函数说明文档开始
    在原图上绘制：  # 说明该函数会把检测信息画在原图上
    1. YOLO最终保留框  # 绘制最终选择的 YOLO 目标框
    2. 中心80%直线检测区域  # 绘制中心 80% 的直线检测区域
    3. 检测数值  # 绘制直线检测函数返回的数值
    """  # 函数说明文档结束
    x1, y1, x2, y2 = det["box_int"]  # 取出最终 YOLO 检测框整数坐标
    ix1, iy1, ix2, iy2 = det["inner_box"]  # 取出中心 80% 检测区域整数坐标
    conf = det["conf"]  # 取出该目标框的置信度
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 用绿色绘制最终保留的 YOLO 检测框
    cv2.rectangle(image, (ix1, iy1), (ix2, iy2), (0, 0, 255), 2)  # 用红色绘制中心 80% 的直线检测区域
    if value is None:  # 判断直线检测结果是否为空
        text = f"{side_name}: fail, conf={conf:.3f}"  # 如果检测失败，则显示 fail 和置信度
    else:  # 如果直线检测结果不为空
        try:  # 尝试把检测结果转成浮点数格式化显示
            text = f"{side_name}: {float(value):.4f}, conf={conf:.3f}"  # 数值可转浮点时，保留 4 位小数显示
        except Exception:  # 如果检测结果不能转成浮点数
            text = f"{side_name}: {value}, conf={conf:.3f}"  # 直接按原始字符串或对象格式显示
    text_x = x1  # 设置文字显示的 x 坐标为目标框左上角 x
    text_y = max(30, y1 - 10)  # 设置文字显示的 y 坐标，避免文字超出图像上边界
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # 在原图上写入检测结果文字
# =========================  # 分隔线，用于区分代码模块
# 4. 处理单张图片  # 当前模块用于完整处理一张图片
# =========================  # 分隔线，用于区分代码模块
def process_one_image(image, model, extract_line_value_func, target_class_id=0, yolo_conf=0.15, conf_thres=0.20, inner_ratio=0.8):  # 定义单张图像处理函数
    """  # 函数说明文档开始
    处理单张图片。  # 该函数完成 YOLO 检测、左右筛选、ROI 裁剪、直线检测和结果绘制
    返回:  # 返回值说明
        left_value: 左侧检测数值  # 左侧直线检测结果
        right_value: 右侧检测数值  # 右侧直线检测结果
        result_image: 画好结果的图像  # 带框和数值的结果图
    """  # 函数说明文档结束
    result_image = image.copy()  # 复制一份原图，用于绘制结果，避免修改原始图像
    results = model(image, conf=yolo_conf)  # 使用 YOLO 模型对当前图像进行检测
    yolo_result = results[0]  # 取出当前图像对应的 YOLO 检测结果
    selected = select_left_right_targets_with_inner_roi(image=image, yolo_result=yolo_result, target_class_id=target_class_id, conf_thres=conf_thres, inner_ratio=inner_ratio)  # 筛选左右目标并裁剪中心 80% 区域
    left_value = None  # 初始化左侧检测结果为空
    right_value = None  # 初始化右侧检测结果为空
    if selected["left"] is not None:  # 判断左侧目标是否存在
        left_line_roi = selected["left"]["line_roi"]  # 取出左侧目标中心 80% 的直线检测区域
        try:  # 尝试执行左侧直线检测
            left_value = extract_line_value_func(left_line_roi)  # 调用用户传入的直线检测函数，得到左侧检测数值
        except Exception as e:  # 捕获左侧直线检测过程中的异常
            print("左侧直线检测失败:", e)  # 打印左侧直线检测失败原因
            left_value = None  # 左侧检测失败时，将结果设置为空
        draw_one_result(result_image, selected["left"], side_name="left", value=left_value)  # 把左侧检测框、内部框和检测结果画到结果图上
    else:  # 如果左侧目标不存在
        cv2.putText(result_image, "left: not found", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)  # 在结果图上提示左侧目标未找到
    if selected["right"] is not None:  # 判断右侧目标是否存在
        right_line_roi = selected["right"]["line_roi"]  # 取出右侧目标中心 80% 的直线检测区域
        try:  # 尝试执行右侧直线检测
            right_value = extract_line_value_func(right_line_roi)  # 调用用户传入的直线检测函数，得到右侧检测数值
        except Exception as e:  # 捕获右侧直线检测过程中的异常
            print("右侧直线检测失败:", e)  # 打印右侧直线检测失败原因
            right_value = None  # 右侧检测失败时，将结果设置为空
        draw_one_result(result_image, selected["right"], side_name="right", value=right_value)  # 把右侧检测框、内部框和检测结果画到结果图上
    else:  # 如果右侧目标不存在
        cv2.putText(result_image, "right: not found", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)  # 在结果图上提示右侧目标未找到
    h, w = image.shape[:2]  # 获取图像高度和宽度
    cv2.line(result_image, (w // 2, 0), (w // 2, h), (255, 0, 0), 2)  # 在图像中心位置画一条蓝色竖线，用于检查左右划分是否正确
    return left_value, right_value, result_image  # 返回左侧数值、右侧数值和画好结果的图像
# =========================  # 分隔线，用于区分代码模块
# 5. CSV写入  # 当前模块用于创建 CSV 文件和追加检测结果
# =========================  # 分隔线，用于区分代码模块
def write_csv_header(csv_path):  # 定义写入 CSV 表头的函数
    """  # 函数说明文档开始
    创建CSV并写入表头。  # 该函数会新建或覆盖 CSV 文件，并写入列名
    """  # 函数说明文档结束
    with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:  # 以写入模式打开 CSV 文件，utf-8-sig 可以让 Excel 正确识别中文
        writer = csv.writer(f)  # 创建 CSV 写入器对象
        writer.writerow(["图像名称", "左侧检测结果", "右侧检测结果"])  # 写入 CSV 表头
def append_result_to_csv(csv_path, image_name, left_value, right_value):  # 定义追加写入单张图片结果的函数
    """  # 函数说明文档开始
    追加单张图片检测结果。  # 该函数每处理完一张图片，就往 CSV 增加一行
    """  # 函数说明文档结束
    with open(csv_path, mode="a", newline="", encoding="utf-8-sig") as f:  # 以追加模式打开 CSV 文件
        writer = csv.writer(f)  # 创建 CSV 写入器对象
        writer.writerow([image_name, "" if left_value is None else left_value, "" if right_value is None else right_value])  # 写入图像名称、左侧结果、右侧结果
# =========================  # 分隔线，用于区分代码模块
# 6. 递归处理文件夹  # 当前模块用于处理输入文件夹及其所有子文件夹中的图片
# =========================  # 分隔线，用于区分代码模块
def process_folder(input_dir, output_dir, model_path, extract_line_value_func, target_class_id=0, yolo_conf=0.15, conf_thres=0.20, inner_ratio=0.8):  # 定义文件夹批量处理函数
    """  # 函数说明文档开始
    递归处理 input_dir 及其子文件夹下的所有图片。  # 说明该函数会遍历所有子目录
    输出:  # 输出内容说明
        output_dir/detect_result.csv  # CSV 检测结果文件
        output_dir/result_images/  # 画框后的结果图文件夹
    """  # 函数说明文档结束
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")  # 定义支持处理的图片格式后缀
    os.makedirs(output_dir, exist_ok=True)  # 创建输出总文件夹，如果已存在则不报错
    result_image_dir = os.path.join(output_dir, "result_images")  # 构造结果图保存文件夹路径
    os.makedirs(result_image_dir, exist_ok=True)  # 创建结果图保存文件夹，如果已存在则不报错
    csv_path = os.path.join(output_dir, "detect_result.csv")  # 构造 CSV 结果文件路径
    write_csv_header(csv_path)  # 创建 CSV 文件并写入表头
    model = YOLO(model_path)  # 加载 YOLO 模型权重文件
    total_count = 0  # 初始化图片总数量计数器
    success_count = 0  # 初始化成功处理图片数量计数器
    for root, dirs, files in os.walk(input_dir):  # 递归遍历输入文件夹及所有子文件夹
        for filename in files:  # 遍历当前文件夹下的所有文件名
            if not filename.lower().endswith(image_extensions):  # 判断当前文件是否不是支持的图片格式
                continue  # 如果不是图片，则跳过
            total_count += 1  # 图片总数量加 1
            image_path = os.path.join(root, filename)  # 拼接当前图片的完整路径
            relative_path = os.path.relpath(image_path, input_dir)  # 计算当前图片相对于输入文件夹的相对路径
            relative_dir = os.path.dirname(relative_path)  # 获取当前图片所在的相对子文件夹路径
            save_sub_dir = os.path.join(result_image_dir, relative_dir)  # 构造对应的结果图输出子文件夹路径
            os.makedirs(save_sub_dir, exist_ok=True)  # 创建结果图输出子文件夹
            name, ext = os.path.splitext(filename)  # 分离图片文件名和扩展名
            save_image_name = name + "_result" + ext  # 构造结果图文件名，在原文件名后添加 _result
            save_image_path = os.path.join(save_sub_dir, save_image_name)  # 构造结果图完整保存路径
            image = cv2.imread(image_path)  # 使用 OpenCV 读取当前图片
            if image is None:  # 判断图像是否读取失败
                print("图像读取失败:", image_path)  # 打印读取失败的图片路径
                append_result_to_csv(csv_path=csv_path, image_name=relative_path, left_value=None, right_value=None)  # 在 CSV 中记录该图片结果为空
                continue  # 跳过当前图片，继续处理下一张
            print("正在处理:", relative_path)  # 打印当前正在处理的图片相对路径
            try:  # 尝试处理当前图片
                left_value, right_value, result_image = process_one_image(image=image, model=model, extract_line_value_func=extract_line_value_func, target_class_id=target_class_id, yolo_conf=yolo_conf, conf_thres=conf_thres, inner_ratio=inner_ratio)  # 调用单张图片处理函数
                cv2.imwrite(save_image_path, result_image)  # 将画好框和数值的结果图保存到指定路径
                append_result_to_csv(csv_path=csv_path, image_name=relative_path, left_value=left_value, right_value=right_value)  # 将当前图片的左右检测结果写入 CSV
                success_count += 1  # 成功处理图片数量加 1
                print("完成:", relative_path, "左侧:", left_value, "右侧:", right_value)  # 打印当前图片处理完成信息
            except Exception as e:  # 捕获当前图片处理过程中出现的异常
                print("处理失败:", image_path)  # 打印处理失败的图片路径
                print("错误信息:", e)  # 打印具体错误信息
                append_result_to_csv(csv_path=csv_path, image_name=relative_path, left_value=None, right_value=None)  # 在 CSV 中记录该图片结果为空
    print("全部处理完成")  # 打印全部处理完成提示
    print("图片总数:", total_count)  # 打印总图片数量
    print("成功处理:", success_count)  # 打印成功处理数量
    print("结果图保存路径:", result_image_dir)  # 打印结果图保存路径
    print("CSV结果路径:", csv_path)  # 打印 CSV 保存路径
    return csv_path, result_image_dir  # 返回 CSV 文件路径和结果图文件夹路径
# =========================  # 分隔线，用于区分代码模块
# 7. 主函数入口  # 当前模块是程序运行入口
# =========================  # 分隔线，用于区分代码模块
if __name__ == "__main__":  # 判断当前脚本是否作为主程序运行
    input_dir = r"D:\your_input_images"  # 设置输入图片文件夹路径，需要替换成你的实际图片文件夹
    output_dir = r"D:\your_output_results"  # 设置输出结果文件夹路径，需要替换成你的实际保存路径
    model_path = r"D:\your_model\best.pt"  # 设置 YOLO 模型权重路径，需要替换成你的 best.pt 路径
    process_folder(input_dir=input_dir, output_dir=output_dir, model_path=model_path, extract_line_value_func=extract_line_value, target_class_id=0, yolo_conf=0.15, conf_thres=0.20, inner_ratio=0.8)  # 调用批量处理函数，开始处理文件夹下所有图片
