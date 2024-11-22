import cv2
import os
import numpy as np
import time
import subprocess


SCALE = 2.4
isReserve = True  # Mask 反转
isErode = True  # Mask 膨胀
isLarge = True  # Mask 拉伸


class Disp2Stereo:
    def __init__(self, orig_path, disp_path):
        self.orig_img = cv2.imread(orig_path, cv2.IMREAD_ANYCOLOR)
        self.disp_img = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)  # 读取16位视差图

        if self.orig_img is None or self.disp_img is None:
            print(f"origPath = {orig_path}")
            print(f"dispPath = {disp_path}")
            return

        self.left_img = None
        self.right_img = None
        self.left_msk = None
        self.right_msk = None
        self.left_large = 0.2
        self.right_large = 0.2
        self.erode_element = (15, 15)
        self.erode_element1 = (3, 3)

    def stereo_left(self):
        self.left_img = np.full_like(self.orig_img, 255)  # 初始化为白色图像
        self.left_msk = np.zeros_like(self.orig_img)  # 初始化为黑色Mask

        max_value = 0
        new_point_x = []
        new_point_y = []

        for y in range(self.orig_img.shape[0]):
            for x in range(self.orig_img.shape[1] - 1, -1, -1):
                disp_value = self.disp_img[y, x]
                max_value = max(max_value, int(disp_value / SCALE))

                left_x1 = x + int(np.floor(disp_value / SCALE))
                left_x2 = x + int(np.ceil(disp_value / SCALE))

                if 0 <= left_x1 < self.orig_img.shape[1]:
                    self.left_img[y, left_x1] = self.orig_img[y, x]
                    self.left_msk[y, left_x1] = [255, 255, 255]

                if 0 <= left_x2 < self.orig_img.shape[1]:
                    self.left_img[y, left_x2] = self.orig_img[y, x]
                    self.left_msk[y, left_x2] = [255, 255, 255]

        print(f"maxDispValue: {max_value} 像素")

        if isErode:
            element = cv2.getStructuringElement(cv2.MORPH_RECT, self.erode_element)
            element1 = cv2.getStructuringElement(cv2.MORPH_RECT, self.erode_element1)
            self.left_msk = cv2.erode(self.left_msk, element)
            self.left_msk = cv2.dilate(self.left_msk, element)
            for _ in range(3):
                self.left_msk = cv2.GaussianBlur(self.left_msk, (15, 15), 0)
            _, self.left_msk = cv2.threshold(self.left_msk, 128, 255, cv2.THRESH_BINARY)
            self.left_msk = cv2.erode(self.left_msk, element1)

        if isLarge:
            for y in range(self.left_msk.shape[0]):
                for x in range(self.left_msk.shape[1] - 1, -1, -1):
                    if np.all(self.left_msk[y, x] == [0, 0, 0]):
                        disp_value = self.disp_img[y, x]
                        for i in range(int(self.left_large * disp_value)):
                            new_x = x - i
                            if new_x >= 0:
                                new_point_x.append(new_x)
                                new_point_y.append(y)

            for i in range(len(new_point_x)):
                self.left_msk[new_point_y[i], new_point_x[i]] = [0, 0, 0]

    def stereo_right(self):
        self.right_img = np.full_like(self.orig_img, 255)  # 初始化为白色图像
        self.right_msk = np.zeros_like(self.orig_img)  # 初始化为黑色Mask

        new_point_x = []
        new_point_y = []

        for y in range(self.orig_img.shape[0]):
            for x in range(self.orig_img.shape[1]):
                disp_value = self.disp_img[y, x]

                right_x1 = x - int(np.floor(disp_value / SCALE))
                right_x2 = x - int(np.ceil(disp_value / SCALE))

                if 0 <= right_x1 < self.orig_img.shape[1]:
                    self.right_img[y, right_x1] = self.orig_img[y, x]
                    self.right_msk[y, right_x1] = [255, 255, 255]

                if 0 <= right_x2 < self.orig_img.shape[1]:
                    self.right_img[y, right_x2] = self.orig_img[y, x]
                    self.right_msk[y, right_x2] = [255, 255, 255]

        if isErode:
            element = cv2.getStructuringElement(cv2.MORPH_RECT, self.erode_element)
            element1 = cv2.getStructuringElement(cv2.MORPH_RECT, self.erode_element1)
            self.right_msk = cv2.erode(self.right_msk, element)
            self.right_msk = cv2.dilate(self.right_msk, element)
            for _ in range(3):
                self.right_msk = cv2.GaussianBlur(self.right_msk, (15, 15), 0)
            _, self.right_msk = cv2.threshold(self.right_msk, 128, 255, cv2.THRESH_BINARY)
            self.right_msk = cv2.erode(self.right_msk, element1)

        if isLarge:
            for y in range(self.right_msk.shape[0]):
                for x in range(self.right_msk.shape[1]):
                    if np.all(self.right_msk[y, x] == [0, 0, 0]):
                        disp_value = self.disp_img[y, x]
                        for i in range(int(self.right_large * disp_value)):
                            new_x = x + i
                            if new_x < self.right_msk.shape[1]:
                                new_point_x.append(new_x)
                                new_point_y.append(y)

            for i in range(len(new_point_x)):
                self.right_msk[new_point_y[i], new_point_x[i]] = [0, 0, 0]

        if isReserve:
            self.left_msk = cv2.bitwise_not(self.left_msk)
            self.right_msk = cv2.bitwise_not(self.right_msk)

    def write_img(self, dst_path):
        color_folder = os.path.join(dst_path, "image/")
        #resize: fill with wanted pixels
        self.left_img = cv2.resize(self.left_img, (512, 512))
        self.right_img = cv2.resize(self.right_img, (512, 512)) 
        os.makedirs(color_folder, exist_ok=True)
        cv2.imwrite(os.path.join(color_folder, "left.png"), self.left_img)
        cv2.imwrite(os.path.join(color_folder, "right.png"), self.right_img)

        # 获取图像的尺寸 (高度, 宽度)
        height, width = self.left_img.shape[:2]
        # 调整图像大小，将宽度和高度缩小为原来的四分之一
        left_image_s = cv2.resize(self.left_img, (width // 4, height // 4), cv2.INTER_AREA)
        right_image_s = cv2.resize(self.right_img, (width // 4, height // 4), cv2.INTER_AREA)
        #resize: fill with wanted pixels
        left_image_s = cv2.resize(left_image_s, (512, 512))
        right_image_s = cv2.resize(right_image_s, (512, 512)) 
        # 保存调整后的图像
        cv2.imwrite(os.path.join(color_folder, "left_small.png"), left_image_s)
        # 保存调整后的图像
        cv2.imwrite(os.path.join(color_folder, "right_small.png"), right_image_s)

        print(f"彩色图像保存成功: {color_folder}")

    def write_mask(self, dst_path):
        mask_folder = os.path.join(dst_path, "mask/")
        #resize: fill with wanted pixels
        self.left_msk = cv2.resize(self.left_msk, (512, 512))
        self.right_msk = cv2.resize(self.right_msk, (512, 512)) 
        os.makedirs(mask_folder, exist_ok=True)
        cv2.imwrite(os.path.join(mask_folder, "left.png"), self.left_msk)
        cv2.imwrite(os.path.join(mask_folder, "right.png"), self.right_msk)
        print(f"Mask图像保存成功: {mask_folder}")




def main(origpict, disppict, savepict,step):
    if not savepict.endswith('/'):
        savepict += '/'
    start_time = time.time()

    if step == 0:
        # 左右视差图
        image_obj = Disp2Stereo(origpict, disppict)
        image_obj.stereo_left()
        image_obj.stereo_right()
        end_time = time.time()

        print(f"time_used: {(end_time - start_time) * 1000:.2f} ms")

        # 保存左右mask图
        image_obj.write_img(savepict)
        image_obj.write_mask(savepict)

        print("Step 1 fished successfully!/n")






if __name__ == "__main__":
    import sys

    print("Usage: python script.py <orig_image> <disp_image> <save_path> <step>")

    if len(sys.argv) == 5:
        #sys.exit(1)
        orig_image = sys.argv[1]
        disp_image = sys.argv[2]
        save_path = sys.argv[3]
        step = int(sys.argv[4])
    else:
        orig_image = './input/03.png'
        disp_image = './input/03_depth.png'
        save_path = './output'
        step = 0


    main(orig_image, disp_image, save_path,step)
