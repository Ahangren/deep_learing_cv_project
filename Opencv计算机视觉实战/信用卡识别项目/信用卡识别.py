# 导入库
import cv2
import numpy as np
import imutils
from imutils import contours  # 轮廓处理工具
from utils import load_template_digits  # 导入自定义工具函数


def recognize_credit_card(image_path):
    """识别信用卡号码的主函数"""

    # ========== 1. 图像预处理 ==========
    # 读取信用卡图像
    image = cv2.imread(image_path)
    # 调整宽度为300像素（保持比例）
    image = imutils.resize(image, width=300)
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ========== 2. 定位数字区域 ==========
    # 定义形态学操作的内核
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))  # 矩形内核（宽9高3）
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 正方形内核

    # 顶帽操作（突出亮色区域：信用卡数字通常是亮的）
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

    # Sobel边缘检测（x方向梯度）
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)  # 取绝对值
    # 归一化到0-255范围
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
    gradX = gradX.astype("uint8")  # 转换为8位无符号整数

    # 闭操作（连接数字区域）
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

    # 二值化（Otsu自动阈值）
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # 再次闭操作（填充小孔洞）
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    # ========== 3. 查找数字组轮廓 ==========
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)  # 兼容不同OpenCV版本
    locs = []  # 存储数字组位置

    # 遍历所有轮廓
    for (i, c) in enumerate(cnts):
        print("找到了吗")
        # 获取轮廓的外接矩形
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)  # 计算宽高比

        # 筛选可能是数字组的区域（基于宽高比和大小）
        if ar > 2.5 and ar < 4.0:  # 信用卡数字组的典型宽高比
            if (w > 40 and w < 55) and (h > 10 and h < 20):  # 典型大小
                locs.append((x, y, w, h))  # 记录合格区域

    # 按x坐标排序（从左到右）
    locs = sorted(locs, key=lambda x: x[0])

    # ========== 4. 加载数字模板 ==========
    digits = load_template_digits()  # 调用工具函数

    # ========== 5. 识别每个数字 ==========
    output = []  # 存储识别结果

    # 遍历每个数字组区域
    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        # 提取数字组区域（扩大5像素边界）
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        # 二值化
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # 查找组内每个数字的轮廓
        digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digitCnts = imutils.grab_contours(digitCnts)
        # 按从左到右排序
        digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

        groupOutput = []  # 存储当前组的识别结果

        # 遍历组内每个数字
        for c in digitCnts:
            # 获取数字边界框
            (x, y, w, h) = cv2.boundingRect(c)
            # 提取数字区域
            roi = group[y:y + h, x:x + w]
            # 调整大小与模板匹配
            roi = cv2.resize(roi, (57, 88))

            # 模板匹配（比较当前数字与所有模板）
            scores = []
            for (digit, digitROI) in digits.items():
                # 计算相似度（相关系数）
                result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)

            # 选择最高分对应的数字
            groupOutput.append(str(np.argmax(scores)))

        # 在图像上绘制结果
        # 画矩形框
        cv2.rectangle(image, (gX - 5, gY - 5),
                      (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
        # 显示识别结果
        cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        output.extend(groupOutput)  # 添加到总结果

    # ========== 6. 输出结果 ==========
    print("识别到的信用卡号码: {}".format("".join(output)))
    cv2.imshow("Credit Card Recognition", image)
    cv2.waitKey(0)  # 等待按键关闭窗口


if __name__ == "__main__":
    # 示例：识别测试图像
    recognize_credit_card("images/img_2.png")