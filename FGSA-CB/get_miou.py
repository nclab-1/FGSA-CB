# import os
#
# from PIL import Image
# from tqdm import tqdm
#
# from deeplab import DeeplabV3
# from utils.utils_metrics import compute_mIoU, show_results
#
# '''
# 进行指标评估需要注意以下几点：
# 1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
# 2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
# '''
# if __name__ == "__main__":
#     #---------------------------------------------------------------------------#
#     #   miou_mode用于指定该文件运行时计算的内容
#     #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
#     #   miou_mode为1代表仅仅获得预测结果。
#     #   miou_mode为2代表仅仅计算miou。
#     #---------------------------------------------------------------------------#
#     miou_mode       = 0
#     #------------------------------#
#     #   分类个数+1、如2+1
#     #------------------------------#
#     num_classes     = 6
#     #--------------------------------------------#
#     #   区分的种类，和json_to_dataset里面的一样
#     #--------------------------------------------#
#     name_classes    = ["_background_", "Impervious surfaces", "Car", "Tree", "Low vegetation", "Building"]
#     # name_classes    = ["_background_","cat","dog"]
#     #-------------------------------------------------------#
#     #   指向VOC数据集所在的文件夹
#     #   默认指向根目录下的VOC数据集
#     #-------------------------------------------------------#
#     VOCdevkit_path  ='VOCdevkit'   #P_IRRG
#
#     image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/test.txt"),'r').read().splitlines()
#     gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
#     miou_out_path   = "miou_out"
#     pred_dir        = os.path.join(miou_out_path, 'detection-results')
#
#     if miou_mode == 0 or miou_mode == 1:
#         if not os.path.exists(pred_dir):
#             os.makedirs(pred_dir)
#
#         print("Load model.")
#         deeplab = DeeplabV3()
#         print("Load model done.")
#
#         print("Get predict result.")
#         for image_id in tqdm(image_ids):
#             image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".png")
#             image       = Image.open(image_path)
#             image       = deeplab.get_miou_png(image)
#             image.save(os.path.join(pred_dir, image_id + ".png"))
#         print("Get predict result done.")
#
#     if miou_mode == 0 or miou_mode == 2:
#         print("Get miou.")
#         hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
#         print("Get miou done.")
#         show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)


import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from deeplab import DeeplabV3


def colorize_segmentation(result_array, color_map):
    """将类别分割结果转化为彩色示意图，仅映射类别范围内的像素"""
    color_result = np.zeros((result_array.shape[0], result_array.shape[1], 3), dtype=np.uint8)
    for label, color in color_map.items():
        mask = (result_array == label)
        color_result[mask] = color
    return Image.fromarray(color_result)


if __name__ == "__main__":
    # 设置为1，表示仅生成预测结果并生成颜色分割图
    miou_mode = 1
    num_classes = 6
    name_classes = ["_background_", "Impervious surfaces", "Car", "Tree", "Low vegetation", "Building"]
    VOCdevkit_path = 'VOCdevkit'

    # 读取测试集图像ID
    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/test.txt"), 'r').read().splitlines()
    pred_dir = os.path.join("miou_out", 'detection-results')
    color_pred_dir = os.path.join("miou_out", 'color-detection-results')

    # 颜色映射：定义每个标签的颜色
    color_map = {
        0: (255, 0, 0),       # 红色：_background_
        1: (255, 255, 255),   # 白色：Impervious surfaces
        2: (255, 255, 0),     # 黄色：Car
        3: (0, 255, 0),       # 绿色：Tree
        4: (0, 255, 255),     # 青色：Low vegetation
        5: (0, 0, 255)        # 蓝色：Building
    }

    # 创建输出文件夹
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(color_pred_dir, exist_ok=True)

    print("Loading model...")
    deeplab = DeeplabV3()
    print("Model loaded.")

    print("Generating segmentation results...")
    for image_id in tqdm(image_ids):
        image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages", f"{image_id}.png")
        image = Image.open(image_path)

        # 获取类别分割的灰度结果
        gray_result = deeplab.get_miou_png(image)
        gray_result.save(os.path.join(pred_dir, f"{image_id}.png"))

        # 将灰度分割图转换为彩色图
        result_array = np.array(gray_result)
        color_result = colorize_segmentation(result_array, color_map)
        color_result.save(os.path.join(color_pred_dir, f"{image_id}_color.png"))

    print("Segmentation results generated.")

