#include "opencv4/opencv2/core/cvdef.h"
#include "opencv4/opencv2/core/hal/interface.h"
#include "opencv4/opencv2/core/mat.hpp"
#include "opencv4/opencv2/highgui.hpp"
#include <filesystem>   // 引入 C++17 文件系统头文件
#include <iostream>     // 引入标准输入输出流头文件
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/opencv.hpp>   // 引入 OpenCV 头文件
#include <string>               // 引入 STL 字符串头文件
#include <vector>               // 引入 STL 向量头文件

using namespace cv;    // 使用 OpenCV 命名空间
using namespace std;   // 使用标准命名空间

#define DEPTH_TYPE uchar

const float SCALE     = 1.0f;    // 设定缩放因子为 2.4
const bool  isReserve = true;    // 是否保留掩码的布尔标志
const bool  isErode   = true;    // 是否进行腐蚀操作的布尔标志
const bool  isLarge   = false;   // 是否进行大范围填充的布尔标志
// const float zoomratio = 8;    // 设定推理缩放因子为 4
const int BOARD  = 50;    //
const int WIDTH  = 512;   // resize
const int HEIGHT = 512;

// 滑动窗口操作（2D） Sliding window algorithm
cv::Mat slidingWindow2D(const cv::Mat& input, int window_size, int stride) {
    CV_Assert(input.dims == 2);   // 确保输入为二维数组
    int     rows = (input.rows - window_size) / stride + 1;
    int     cols = (input.cols - window_size) / stride + 1;
    cv::Mat output(rows, cols, input.type(), cv::Scalar::all(0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cv::Range row_range(i * stride, i * stride + window_size);
            cv::Range col_range(j * stride, j * stride + window_size);
            cv::Mat   window = input(row_range, col_range);
            double    min_val, max_val;
            cv::minMaxLoc(window, &min_val, &max_val);
            output.at<uchar>(i, j) = static_cast<uchar>(max_val);
        }
    }
    return output;
}

// 最大池化 the biggest pool
cv::Mat maxPool2D(const cv::Mat& input, int kernel_size, int stride, int padding = 0) {
    cv::Mat padded;
    if (padding > 0) {
        cv::copyMakeBorder(input, padded, padding, padding, padding, padding, cv::BORDER_CONSTANT, 0);
    } else {
        padded = input;
    }
    return slidingWindow2D(padded, kernel_size, stride);
}

// 边缘检测与深度区域提取
cv::Mat depthEdge(const cv::Mat& depth, int atol = -1, float rtol = -1.0, int kernel_size = 3) {
    CV_Assert(depth.type() == CV_8U);   // 确保输入为灰度图
                                        //esnure the input are grayscale images

    cv::Mat maxDepth, minDepth;

    maxDepth = maxPool2D(depth, kernel_size, 1, kernel_size / 2);
    minDepth = maxPool2D(255 - depth, kernel_size, 1, kernel_size / 2);
    minDepth = 255 - minDepth;


    cv::Mat diff = maxDepth - minDepth;
    cv::Mat edge = cv::Mat::zeros(depth.size(), CV_8U);

    if (atol > 0) {
        edge = diff > atol;
    }
    if (rtol > 0) {
        cv::Mat rtolMask;
        cv::divide(diff, depth + 1, rtolMask);   // 逐元素计算 diff / (depth + 1)
                                                //calculate element by element diff / (depth + 1)



        // 生成掩码，保留 diff / (depth + 1) > rtol 的位置
        // Generate mask, retain diff / (depth + 1) > rtol position
        cv::Mat resultMask = (rtolMask > rtol);

        edge |= resultMask >= 1;
    }


    return edge;
}

class Disp2Stereo {
public:
    Mat   orig_img;                        // 原始图像
    Mat   disp_img;                        // 视差图像
    Mat   orig_img_inpaint;                // 原始图像
    Mat   disp_img_inpaint;                // 视差图像
    Mat   edge_mask;                       // 边缘掩码
    Mat   left_img, right_img;             // 左视图像和右视图像
    Mat   left_msk, right_msk;             // 左视图像和右视图像的掩码
    float left_large     = 0.8;            // 左视图像大范围填充的比例
    float right_large    = 0.8;            // 右视图像大范围填充的比例
    Size  erode_element  = Size(15, 15);   // 腐蚀操作的元素大小 Size(15, 15);
    Size  erode_element1 = Size(15, 15);   // 腐蚀操作的另一元素大小 Size(3, 3);
    Size  gselement      = Size(9, 9);     // 高斯模糊的元素大小

    Disp2Stereo(const string& orig_path, const string& disp_path, const string& orig_path_inpaint, const string& disp_path_inpaint) {
        orig_img         = imread(orig_path, IMREAD_COLOR);               // 读取原始图像
        disp_img         = imread(disp_path, IMREAD_GRAYSCALE);           // 读取视差图像
        orig_img_inpaint = imread(orig_path_inpaint, IMREAD_COLOR);       // 读取原始图像
        disp_img_inpaint = imread(disp_path_inpaint, IMREAD_GRAYSCALE);   // 读取视差图像
        edge_mask        = cv::Mat::zeros(orig_img.size(), CV_8U);
        // edge_mask        = depthEdge(disp_img, 10, -0.001, 15);

        imwrite("./edge_mask.png", edge_mask);

        if (orig_img.empty() || disp_img.empty() || orig_img_inpaint.empty() || disp_img_inpaint.empty()) {   // 如果图像读取失败
            cout << "Error loading images: " << orig_path << ", " << disp_path << ", " << orig_path_inpaint << ", " << disp_path_inpaint
                 << endl;   // 输出错误信息
        }
        cout << "=-================" << disp_img.type() << endl;
    }

    void stereoLeft() {
        left_img = Mat(orig_img.size(), orig_img.type(), Scalar(255, 255, 255));   // 创建与原始图像同尺寸的白色左视图像
        left_msk = Mat(orig_img.size(), orig_img.type(), Scalar(0, 0, 0));         // 创建与原始图像同尺寸的黑色左掩码

        int         max_value1 = 0, max_value2 = 0;   // 记录视差图像的最大值
        vector<int> new_point_x, new_point_y;         // 存储新像素坐标的向量


        for (int y = 0; y < orig_img.rows; ++y) {            // 遍历原始图像的每一行
            for (int x = orig_img.cols - 1; x >= 0; --x) {   // 从右到左遍历每一列

                if (edge_mask.at<uchar>(y, x)) continue;                // hjh add
                int disp_value = 255 - disp_img.at<DEPTH_TYPE>(y, x);   // 获取当前像素的视差值

                int left_x1 = x + floor(disp_value * SCALE);   // 计算左视图像中的第一个新 x 坐标
                int left_x2 = left_x1 + 1;                     // 计算左视图像中的第二个新 x 坐标

                if (left_x1 >= 0 && left_x1 < orig_img.cols) {                   // 如果第一个新 x 坐标在图像范围内
                    left_img.at<Vec3b>(y, left_x1) = orig_img.at<Vec3b>(y, x);   // 将原图像的像素复制到左视图像
                    left_msk.at<Vec3b>(y, left_x1) = Vec3b(255, 255, 255);       // 将掩码的像素设置为白色
                }
                if (left_x2 >= 0 && left_x2 < orig_img.cols) {                   // 如果第二个新 x 坐标在图像范围内
                    left_img.at<Vec3b>(y, left_x2) = orig_img.at<Vec3b>(y, x);   // 将原图像的像素复制到左视图像
                    left_msk.at<Vec3b>(y, left_x2) = Vec3b(255, 255, 255);       // 将掩码的像素设置为白色
                }
            }
        }

        if (isErode) {                                                             // 如果需要进行腐蚀操作
            Mat element  = getStructuringElement(MORPH_ELLIPSE, erode_element);    // 创建腐蚀操作的结构元素
            Mat element1 = getStructuringElement(MORPH_ELLIPSE, erode_element1);   // 创建另一种腐蚀操作的结构元素
            erode(left_msk, left_msk, element);                                    // 对掩码进行腐蚀操作
            dilate(left_msk, left_msk, element1);                                  // 对掩码进行膨胀操作
            for (int i = 0; i < 3; ++i) {                                          // 进行高斯模糊处理
                GaussianBlur(left_msk, left_msk, gselement, 0);                    // 高斯模糊
            }
            threshold(left_msk, left_msk, 128, 255,
                      THRESH_BINARY);              // 对掩码进行阈值处理
            erode(left_msk, left_msk, element1);   // 再次对掩码进行腐蚀操作
        }

        if (isLarge) {                                                                     // 如果需要进行大范围填充
            for (int y = 0; y < left_msk.rows; ++y) {                                      // 遍历掩码的每一行
                for (int x = left_msk.cols - 1; x >= 0; --x) {                             // 从右到左遍历每一列
                    if (left_msk.at<Vec3b>(y, x) == Vec3b(0, 0, 0)) {                      // 如果掩码值为黑色
                        int disp_value = 255 - disp_img.at<DEPTH_TYPE>(y, x);              // 获取视差值
                        for (int i = 0; i < int(left_large * disp_value * SCALE); ++i) {   // 根据视差值进行填充
                            int new_x = x - i;                                             // 计算新的 x 坐标
                            if (new_x >= 0) {                                              // 如果新的 x 坐标在图像范围内
                                new_point_x.push_back(new_x);                              // 存储新的 x 坐标
                                new_point_y.push_back(y);                                  // 存储新的 y 坐标
                            }
                        }
                    }
                }
            }
            for (size_t i = 0; i < new_point_x.size(); ++i) {                          // 更新掩码图像
                left_msk.at<Vec3b>(new_point_y[i], new_point_x[i]) = Vec3b(0, 0, 0);   // 将填充区域的像素值设为黑色
            }
        }
    }

    void stereoRight() {
        right_img = Mat(orig_img.size(), orig_img.type(), Scalar(255, 255, 255));   // 创建与原始图像同尺寸的白色右视图像
        right_msk = Mat(orig_img.size(), orig_img.type(), Scalar(0, 0, 0));         // 创建与原始图像同尺寸的黑色右掩码

        vector<int> new_point_x, new_point_y;   // 存储新像素坐标的向量

        for (int y = 0; y < orig_img.rows; ++y) {                       // 遍历原始图像的每一行
            for (int x = 0; x < orig_img.cols; ++x) {                   // 从左到右遍历每一列
                if (edge_mask.at<uchar>(y, x)) continue;                // hjh add
                int disp_value = 255 - disp_img.at<DEPTH_TYPE>(y, x);   // 获取当前像素的视差值

                int right_x1 = x - floor(disp_value * SCALE);   // 计算右视图像中的第一个新 x 坐标
                int right_x2 = right_x1 - 1;                    // 计算右视图像中的第二个新 x 坐标

                if (right_x1 >= 0 && right_x1 < orig_img.cols) {                   // 如果第一个新 x 坐标在图像范围内
                    right_img.at<Vec3b>(y, right_x1) = orig_img.at<Vec3b>(y, x);   // 将原图像的像素复制到右视图像
                    right_msk.at<Vec3b>(y, right_x1) = Vec3b(255, 255, 255);       // 将掩码的像素设置为白色
                }
                if (right_x2 >= 0 && right_x2 < orig_img.cols) {                   // 如果第二个新 x 坐标在图像范围内
                    right_img.at<Vec3b>(y, right_x2) = orig_img.at<Vec3b>(y, x);   // 将原图像的像素复制到右视图像
                    right_msk.at<Vec3b>(y, right_x2) = Vec3b(255, 255, 255);       // 将掩码的像素设置为白色
                }
            }
        }

        if (isErode) {                                                             // 如果需要进行腐蚀操作
            Mat element  = getStructuringElement(MORPH_ELLIPSE, erode_element);    // 创建腐蚀操作的结构元素
            Mat element1 = getStructuringElement(MORPH_ELLIPSE, erode_element1);   // 创建另一种腐蚀操作的结构元素
            erode(right_msk, right_msk, element);                                  // 对掩码进行腐蚀操作
            dilate(right_msk, right_msk, element1);                                // 对掩码进行膨胀操作
            for (int i = 0; i < 3; ++i) {                                          // 进行高斯模糊处理
                GaussianBlur(right_msk, right_msk, gselement, 0);                  // 高斯模糊
            }
            threshold(right_msk, right_msk, 128, 255,
                      THRESH_BINARY);                // 对掩码进行阈值处理
            erode(right_msk, right_msk, element1);   // 再次对掩码进行腐蚀操作
        }

        if (isLarge) {                                                                      // 如果需要进行大范围填充
            for (int y = 0; y < right_msk.rows; ++y) {                                      // 遍历掩码的每一行
                for (int x = 0; x < right_msk.cols; ++x) {                                  // 从左到右遍历每一列
                    if (right_msk.at<Vec3b>(y, x) == Vec3b(0, 0, 0)) {                      // 如果掩码值为黑色
                        int disp_value = 255 - disp_img.at<DEPTH_TYPE>(y, x);               // 获取视差值
                        for (int i = 0; i < int(right_large * disp_value * SCALE); ++i) {   // 根据视差值进行填充
                            int new_x = x + i;                                              // 计算新的 x 坐标
                            if (new_x < right_msk.cols) {                                   // 如果新的 x 坐标在图像范围内
                                new_point_x.push_back(new_x);                               // 存储新的 x 坐标
                                new_point_y.push_back(y);                                   // 存储新的 y 坐标
                            }
                        }
                    }
                }
            }
            for (size_t i = 0; i < new_point_x.size(); ++i) {                           // 更新掩码图像
                right_msk.at<Vec3b>(new_point_y[i], new_point_x[i]) = Vec3b(0, 0, 0);   // 将填充区域的像素值设为黑色
            }
        }

        if (isReserve) {                         // 如果需要反转掩码
            bitwise_not(left_msk, left_msk);     // 反转左掩码
            bitwise_not(right_msk, right_msk);   // 反转右掩码
        }
    }

    void writeImage(const string& dst_path) {
        string color_folder = dst_path + "/image/";
        std::filesystem::create_directories(color_folder);
        imwrite(color_folder + "left.png", left_img);
        imwrite(color_folder + "right.png", right_img);

        // Resize images
        // resize(left_img, left_img, Size(left_img.cols / zoomratio, left_img.rows
        // / zoomratio)); resize(right_img, right_img, Size(right_img.cols /
        // zoomratio, right_img.rows / zoomratio));
        // resize(left_img, left_img, Size(WIDTH, HEIGHT), INTER_AREA);
        // resize(right_img, right_img, Size(WIDTH, HEIGHT), INTER_AREA);

        // imwrite(color_folder + "left_small.png", left_img);
        // imwrite(color_folder + "right_small.png", right_img);
    }

    void writeMask(const string& dst_path) {
        string mask_folder = dst_path + "/mask/";
        std::filesystem::create_directories(mask_folder);
        imwrite(mask_folder + "left.png", left_msk);
        imwrite(mask_folder + "right.png", right_msk);
    }
};

// 辅助函数：查找比给定 x 值大且离其最近的 x 值
int findNextXRight(const std::vector<int>& xCoords, int xValue) {
    if (xCoords.empty()) {
        return -1;
    }
    auto it = std::lower_bound(xCoords.begin(), xCoords.end(), xValue);
    while (it != xCoords.end() && *it - xValue < 10) {
        it++;
    }
    if (it != xCoords.end()) {
        return *it;
    }
    return -1;   // 如果没有找到合适的 x 值
}

// 辅助函数：查找比给定 x 值小且离其最近的 x 值
int findNextXLeft(const std::vector<int>& xCoords, int xValue) {
    if (xCoords.empty()) {
        return -1;
    }
    auto it = std::lower_bound(xCoords.begin(), xCoords.end(), xValue);
    if (it == xCoords.end()) {
        it = xCoords.end() - 1;
    }
    while (it != xCoords.begin() && xValue - *it < 10) {
        it--;
    }
    if (*it < xValue - 10) {
        return *(it);
    } else {
        return -1;   // 如果没有找到合适的 x 值
    }
}
void expandMask(const std::string& maskLeftPath, const std::string& maskRightPath, const std::string& expand_mask_path,
                const std::string& expand_small_mask_path, const std::string& left_or_right_mask, int expand = 50, int savePixel = 20) {

    // 读取掩码图像
    cv::Mat maskLeft  = cv::imread(maskLeftPath, cv::IMREAD_COLOR);
    cv::Mat maskRight = cv::imread(maskRightPath, cv::IMREAD_COLOR);

    if (maskLeft.empty() || maskRight.empty()) {
        std::cerr << "Error loading images!" << std::endl;
        return;
    }

    cv::Mat edgesLeft, edgesRight;
    cv::Canny(maskLeft, edgesLeft, 100, 200);
    cv::Canny(maskRight, edgesRight, 100, 200);
    // debug
    // cv::imwrite("../edgesLeft.png", edgesLeft);
    // cv::imwrite("../edgesRight.png", edgesRight);

    int                             h = maskLeft.rows;
    int                             w = maskLeft.cols;
    std::map<int, std::vector<int>> yToXLeft, yToXRight;

    // 存储掩码1的边缘点
    for (int y = 0; y < h; ++y) {
        break;
        for (int x = BOARD; x < w - BOARD; ++x) {
            if (edgesLeft.at<uchar>(y, x) == 255) {
                yToXLeft[y].push_back(x);
            }
        }
        // sort(yToXLeft[y].begin(), yToXLeft[y].end());
    }

    // 存储掩码2的边缘点
    for (int y = 0; y < h; ++y) {
        break;
        for (int x = BOARD; x < w - BOARD; ++x) {
            if (edgesRight.at<uchar>(y, x) == 255) {
                yToXRight[y].push_back(x);
            }
        }
        // sort(yToXRight[y].begin(), yToXRight[y].end());
    }

    cv::Mat   maskExpand        = (left_or_right_mask == "left") ? maskLeft.clone() : maskRight.clone();
    const int EXPAND_PIEXL      = 300;
    const int EXPAND_PIEXL_PASS = 200;
    if (left_or_right_mask == "left") {
        for (const auto& [y, xCoords] : yToXLeft) {
            for (int x : xCoords) {
                int nearX = findNextXRight(yToXLeft[y], x);
                int endX  = findNextXRight(yToXRight[y], x);
                if (endX == -1) endX = EXPAND_PIEXL + x;

                for (int idx = x; idx < std::min(w, endX + EXPAND_PIEXL_PASS); ++idx) {
                    if (idx + savePixel < w && nearX != -1 && idx + savePixel > nearX) break;
                    maskExpand.at<cv::Vec3b>(y, idx) = cv::Vec3b(255, 255, 255);
                }
            }
        }
    } else if (left_or_right_mask == "right") {
        for (const auto& [y, xCoords] : yToXRight) {
            for (int x : xCoords) {
                int nearX = findNextXLeft(yToXRight[y], x);
                int endX  = findNextXLeft(yToXLeft[y], x);
                // std::cout << " row : " << y << ' ' << " nearX : " << nearX
                //           << "  endX : " << endX << " x : " << x << std::endl;
                if (endX == -1) endX = x - EXPAND_PIEXL;
                for (int idx = x; idx > std::max(0, endX - EXPAND_PIEXL_PASS); --idx) {
                    if (nearX != -1 && idx - savePixel < nearX) break;
                    maskExpand.at<cv::Vec3b>(y, idx) = cv::Vec3b(255, 255, 255);
                }
            }
        }
    }

    // 应用高斯模糊
    cv::Mat blurredImage;
    cv::GaussianBlur(maskExpand, blurredImage, cv::Size(5, 5), 0);

    // 二值化处理
    cv::Mat binaryImage;
    cv::threshold(blurredImage, binaryImage, 128, 255, cv::THRESH_BINARY);

    // 保存扩展后的图像
    cv::imwrite(expand_mask_path, binaryImage);

    // 缩小并保存
    cv::Mat resized_image;
    cv::resize(binaryImage, resized_image, cv::Size(WIDTH, HEIGHT), 0, 0, cv::INTER_AREA);
    cv::imwrite(expand_small_mask_path, resized_image);
}

void expand_double_mask(const std::string& mask_path_1, const std::string& mask_path_2, const std::string& expand_mask_path,
                        const std::string& expand_small_mask_path, const std::string& direction, int expand = 50, int save_pixel = 20) {
    // 读取两个mask
    cv::Mat mask1 = cv::imread(mask_path_1, -1);
    cv::Mat mask2 = cv::imread(mask_path_2, -1);
    cv::Mat res   = cv::Mat::zeros(mask1.size(), mask1.type());

    int h = mask1.rows;
    int w = mask1.cols;

    if (direction == "right") {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < 40; ++x) {
                res.at<cv::Vec3b>(y, x) = mask1.at<cv::Vec3b>(y, x);
            }
            for (int x = 40; x < w - 1; ++x) {
                cv::Vec3b pixel1     = mask1.at<cv::Vec3b>(y, x);
                cv::Vec3b nxt_pixel1 = mask1.at<cv::Vec3b>(y, x + 1);
                if (pixel1 == cv::Vec3b(255, 255, 255) && nxt_pixel1 == cv::Vec3b(0, 0, 0)) {
                    for (int idx = x; idx < std::min(x + expand, w); ++idx) {
                        if (mask2.at<cv::Vec3b>(y, max(0, idx - int(w * 0.04))) == cv::Vec3b(255, 255, 255)) break;
                        if (idx + save_pixel < w && mask1.at<cv::Vec3b>(y, idx + save_pixel) == cv::Vec3b(255, 255, 255)) break;
                        res.at<cv::Vec3b>(y, idx) = cv::Vec3b(255, 255, 255);
                    }
                }
                if (pixel1 == cv::Vec3b(255, 255, 255)) {
                    res.at<cv::Vec3b>(y, x) = pixel1;
                }
            }
        }
    } else if (direction == "left") {
        for (int y = 0; y < h; ++y) {
            for (int x = w - 1; x > w - 40; --x) {
                res.at<cv::Vec3b>(y, x) = mask1.at<cv::Vec3b>(y, x);
            }
            for (int x = w - 40; x > 0; --x) {
                cv::Vec3b pixel1      = mask1.at<cv::Vec3b>(y, x);
                cv::Vec3b prev_pixel1 = mask1.at<cv::Vec3b>(y, x - 1);
                if (pixel1 == cv::Vec3b(255, 255, 255) && prev_pixel1 == cv::Vec3b(0, 0, 0)) {
                    for (int idx = x; idx > std::max(x - expand, 0); --idx) {
                        if (mask2.at<cv::Vec3b>(y, min(idx + int(w * 0.04), w - 1)) == cv::Vec3b(255, 255, 255)) break;
                        if (idx - save_pixel >= 0 && mask1.at<cv::Vec3b>(y, idx - save_pixel) == cv::Vec3b(255, 255, 255)) break;
                        res.at<cv::Vec3b>(y, idx) = cv::Vec3b(255, 255, 255);
                    }
                }
                if (pixel1 == cv::Vec3b(255, 255, 255)) {
                    res.at<cv::Vec3b>(y, x) = pixel1;
                }
            }
        }
    }

    // 保存处理后的mask
    cv::imwrite(expand_mask_path, res);

    // 缩小并保存
    cv::Mat resized_image;
    // cv::resize(res, resized_image, cv::Size(w / zoomratio, h / zoomratio), 0,
    // 0, cv::INTER_AREA);
    cv::resize(res, resized_image, cv::Size(WIDTH, HEIGHT), 0, 0, cv::INTER_AREA);
    cv::imwrite(expand_small_mask_path, resized_image);
}


void expandMaskMajoy(const std::string& maskLeftPath, const std::string& maskRightPath, const std::string& expand_mask_path,
                     const std::string& expand_small_mask_path, const std::string& mask_major_path, const std::string& left_or_right_mask,
                     int expand_min_width = 100) {

    // 读取掩码图像
    cv::Mat maskLeft  = cv::imread(maskLeftPath, cv::IMREAD_GRAYSCALE);
    cv::Mat maskRight = cv::imread(maskRightPath, cv::IMREAD_GRAYSCALE);

    if (maskLeft.empty() || maskRight.empty()) {
        std::cerr << "Error loading images!" << std::endl;
        return;
    }
    // 确保是二值图像，如果不是，可以使用二值化
    threshold(maskLeft, maskLeft, 127, 255, THRESH_BINARY);
    threshold(maskRight, maskRight, 127, 255, THRESH_BINARY);

    Mat maskExpand = Mat(maskLeft.size(), maskLeft.type(), Scalar(0, 0, 0));   // 创建与原始图像同尺寸的
    Mat mask_major = Mat(maskLeft.size(), maskLeft.type(), Scalar(0, 0, 0));   // 创建与原始图像同尺寸的
    if (left_or_right_mask == "left") {
        int max_width         = 0;
        int max_width_point_y = 0;
        int max_width_point_x = 0;

        // 存储连通区域的标签
        Mat labels;
        // 返回连通区域数量
        int num_labels = connectedComponents(maskLeft, labels, 8, CV_32S);

        // imshow("maskLeft", labels * 255);
        // waitKey();

        for (int y = 0; y < maskLeft.rows; ++y) {
            int tem_width = 0;
            for (int x = maskLeft.cols * 0.1; x < maskLeft.cols; ++x) {
                if (maskLeft.at<uchar>(y, x) == 255) {
                    tem_width++;
                } else if (maskLeft.at<uchar>(y, x) == 0) {
                    if (tem_width > max_width) {
                        max_width         = tem_width;
                        max_width_point_y = y;
                        max_width_point_x = x - 3;
                    }
                    tem_width = 0;
                }
            }
        }

        int keylabel = labels.at<int>(max_width_point_y, max_width_point_x);
        for (int y = 0; y < maskLeft.rows; ++y) {
            bool found = false;   // 标记是否找到 keylabel
            for (int x = 0; x < maskLeft.cols; ++x) {
                if (labels.at<int>(y, x) == keylabel) {
                    found                      = true;   // 找到第一个匹配的点
                    mask_major.at<uchar>(y, x) = 255;
                }


                if (found) {
                    maskExpand.at<uchar>(y, x) = 255;   // 从当前点开始，后面的点都设置为255
                }
            }
        }



    } else {
        int max_width         = 0;
        int max_width_point_y = 0;
        int max_width_point_x = 0;

        for (int y = 0; y < maskRight.rows; ++y) {
            int tem_width = 0;
            for (int x = 0; x < maskRight.cols * 0.9; ++x) {
                if (maskRight.at<uchar>(y, x) == 255)
                    tem_width++;
                else if (maskRight.at<uchar>(y, x) == 0) {
                    if (tem_width > max_width) {
                        max_width         = tem_width;
                        max_width_point_y = y;
                        max_width_point_x = x - 3;
                    }
                    tem_width = 0;
                }
            }
        }

        // 存储连通区域的标签
        Mat labels;
        // 返回连通区域数量
        int num_labels = connectedComponents(maskRight, labels, 8, CV_32S);
        int keylabel   = labels.at<int>(max_width_point_y, max_width_point_x);

        for (int y = 0; y < maskRight.rows; ++y) {
            bool found = false;   // 标记是否找到 keylabel
            for (int x = maskRight.cols - 1; x >= 0; --x) {
                if (labels.at<int>(y, x) == keylabel) {
                    found                      = true;   // 找到第一个匹配的点
                    mask_major.at<uchar>(y, x) = 255;
                }

                if (found) {
                    maskExpand.at<uchar>(y, x) = 255;   // 从当前点开始，后面的点都设置为255
                }
            }
        }
    }


    // // 应用高斯模糊
    // cv::Mat blurredImage;
    // cv::GaussianBlur(maskExpand, blurredImage, cv::Size(5, 5), 0);

    // // 二值化处理
    // cv::Mat binaryImage;
    // cv::threshold(blurredImage, binaryImage, 128, 255, cv::THRESH_BINARY);

    // 保存扩展后的图像
    cv::imwrite(expand_mask_path, maskExpand);
    cv::imwrite(mask_major_path, mask_major);

    // 缩小并保存
    cv::Mat resized_image;
    cv::resize(maskExpand, resized_image, cv::Size(WIDTH, HEIGHT), 0, 0, cv::INTER_AREA);
    cv::imwrite(expand_small_mask_path, resized_image);
}

void run_iopaint(const std::string& image_path, const std::string& mask_path, const std::string& output_path) {
    std::string command =
        "conda run -n gen_left_right iopaint run --model=lama --device=cuda --image=" + image_path + " --mask=" + mask_path + " --output=" + output_path;
    int result = system(command.c_str());

    if (result == 0) {
        std::cout << "Inpaint executed successfully!" << std::endl;
    } else {
        std::cerr << "Error inpaint." << std::endl;
    }
}

void main_process(const std::string& origpict, const std::string& disppict, const std::string& origpict_inpaint, const std::string& disppict_inpaint,
                  const std::string& savepict, const string& img_id, int step) {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (step >= 0) {
        // 假设 Disp2Stereo 是用户定义的类，用于生成左右视差图
        Disp2Stereo image_obj(origpict, disppict, origpict_inpaint, disppict_inpaint);
        image_obj.stereoLeft();
        image_obj.stereoRight();

        // 记录时间
        auto                          end_time  = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_used = end_time - start_time;
        std::cout << "Time used: " << time_used.count() * 1000 << " ms" << std::endl;

        image_obj.writeImage(savepict);
        image_obj.writeMask(savepict);
    }

    if (step >= 1) {
        std::string mask_path_1         = savepict + "/mask/left.png";
        std::string mask_path_2         = savepict + "/mask/right.png";
        std::string expand_mask_1       = savepict + "/mask/left_expand.png";
        std::string expand_mask_2       = savepict + "/mask/right_expand.png";
        std::string expand_small_mask_1 = savepict + "/mask/left_expand_small.png";
        std::string expand_small_mask_2 = savepict + "/mask/right_expand_small.png";
        std::string major_mask_1        = savepict + "/mask/left_major.png";
        std::string major_mask_2        = savepict + "/mask/right_major.png";

        // expand_double_mask(mask_path_1, mask_path_2, expand_mask_1,
        // expand_small_mask_1, "right", 8000, 20); expand_double_mask(mask_path_2,
        // mask_path_1, expand_mask_2, expand_small_mask_2, "left", 8000, 20);
        // expandMask(mask_path_1, mask_path_2, expand_mask_1, expand_small_mask_1, "left", 0, 0);
        // expandMask(mask_path_1, mask_path_2, expand_mask_2, expand_small_mask_2, "right", 0, 0);
        expandMaskMajoy(mask_path_1, mask_path_2, expand_mask_1, expand_small_mask_1, major_mask_1, "left");
        expandMaskMajoy(mask_path_1, mask_path_2, expand_mask_2, expand_small_mask_2, major_mask_2, "right");
    }

    if (step >= 2) {
        std::string expand_small_img_1  = savepict + "/image/left_small.png";
        std::string expand_small_img_2  = savepict + "/image/right_small.png";
        std::string expand_small_mask_1 = savepict + "/mask/left_expand_small.png";
        std::string expand_small_mask_2 = savepict + "/mask/right_expand_small.png";
        std::string output_path_1       = savepict + "/inpaint/left_major";
        std::string output_path_2       = savepict + "/inpaint/right_major";


        std::string ori_mask_1       = savepict + "/mask/left.png";
        std::string ori_mask_2       = savepict + "/mask/right.png";
        std::string ori_small_mask_1 = savepict + "/mask/left_small.png";
        std::string ori_small_mask_2 = savepict + "/mask/right_small.png";
        std::string output_path_3    = savepict + "/inpaint/left";
        std::string output_path_4    = savepict + "/inpaint/right";
        cv::Mat     resized_image, tem;
        tem = cv::imread(ori_mask_1, cv::IMREAD_UNCHANGED);
        cv::resize(tem, resized_image, cv::Size(WIDTH, HEIGHT), 0, 0, cv::INTER_AREA);
        cv::imwrite(ori_small_mask_1, resized_image);
        tem = cv::imread(ori_mask_2, cv::IMREAD_UNCHANGED);
        cv::resize(tem, resized_image, cv::Size(WIDTH, HEIGHT), 0, 0, cv::INTER_AREA);
        cv::imwrite(ori_small_mask_2, resized_image);


        run_iopaint(expand_small_img_1, expand_small_mask_1, output_path_1);
        run_iopaint(expand_small_img_2, expand_small_mask_2, output_path_2);
        run_iopaint(expand_small_img_1, ori_small_mask_1, output_path_3);
        run_iopaint(expand_small_img_2, ori_small_mask_2, output_path_4);
    }

    if (step >= 3) {
        std::string left_img_path            = savepict + "image/left.png";
        std::string mask_left_img_path       = savepict + "mask/left.png";
        std::string mask_left_major_img_path = savepict + "mask/left_major.png";
        std::string left_inpaint_small_path  = savepict + "inpaint/left/left_small.png";
        std::string left_inpaint_major_path  = savepict + "inpaint/left_major/left_small.png";

        std::string right_img_path            = savepict + "image/right.png";
        std::string mask_right_img_path       = savepict + "mask/right.png";
        std::string mask_right_major_img_path = savepict + "mask/right_major.png";
        std::string right_inpaint_small_path  = savepict + "inpaint/right/right_small.png";
        std::string right_inpaint_major_path  = savepict + "inpaint/right_major/right_small.png";

        cv::Mat left_img            = cv::imread(left_img_path);
        cv::Mat mask_left_img       = cv::imread(mask_left_img_path, cv::IMREAD_GRAYSCALE);
        cv::Mat mask_left_major_img = cv::imread(mask_left_major_img_path, cv::IMREAD_GRAYSCALE);
        cv::Mat left_inpaint_major  = cv::imread(left_inpaint_major_path);
        cv::Mat left_inpaint_small  = cv::imread(left_inpaint_small_path);

        cv::Mat right_img            = cv::imread(right_img_path);
        cv::Mat mask_right_img       = cv::imread(mask_right_img_path, cv::IMREAD_GRAYSCALE);
        cv::Mat mask_right_major_img = cv::imread(mask_right_major_img_path, cv::IMREAD_GRAYSCALE);
        cv::Mat right_inpaint_major  = cv::imread(right_inpaint_major_path);
        cv::Mat right_inpaint_small  = cv::imread(right_inpaint_small_path);

        // 检查图像是否成功加载
        if (left_img.empty() || mask_left_img.empty() || left_inpaint_small.empty() || left_inpaint_major.empty() || mask_left_major_img.empty() ||
            right_img.empty() || mask_right_img.empty() || right_inpaint_small.empty() || left_inpaint_major.empty() || right_inpaint_major.empty()) {
            std::cerr << "Error: One or more image files could not be loaded." << std::endl;
            std::cerr << "Check paths and file integrity." << std::endl;
            return;
        }

        // 确保所有图像的尺寸匹配
        if (left_inpaint_small.size() != right_inpaint_small.size()) {
            std::cerr << "Error: Inpaint images size mismatch!" << std::endl;
            return;
        }

        // 调整图像大小
        cv::Mat left_inpaint_small_resized, right_inpaint_small_resized, left_inpaint_major_resized, right_inpaint_major_resized;
        // cv::resize(left_inpaint_small, left_inpaint_small_resized,
        // cv::Size(left_inpaint_small.cols * zoomratio, left_inpaint_small.rows *
        // zoomratio), 0, 0, cv::INTER_AREA); cv::resize(right_inpaint_small,
        // right_inpaint_small_resized, cv::Size(right_inpaint_small.cols *
        // zoomratio, right_inpaint_small.rows * zoomratio), 0, 0, cv::INTER_AREA);
        cv::resize(left_inpaint_small, left_inpaint_small_resized, cv::Size(left_img.cols, left_img.rows), 0, 0, cv::INTER_CUBIC);
        cv::resize(right_inpaint_small, right_inpaint_small_resized, cv::Size(right_img.cols, right_img.rows), 0, 0, cv::INTER_CUBIC);
        cv::resize(left_inpaint_major, left_inpaint_major_resized, cv::Size(left_img.cols, left_img.rows), 0, 0, cv::INTER_CUBIC);
        cv::resize(right_inpaint_major, right_inpaint_major_resized, cv::Size(right_img.cols, right_img.rows), 0, 0, cv::INTER_CUBIC);
        // 生成反向掩码
        cv::Mat mask_inv_1, mask_inv_2, mask_inv_3, mask_inv_4;
        cv::bitwise_not(mask_left_img, mask_inv_1);
        cv::bitwise_not(mask_right_img, mask_inv_2);
        cv::bitwise_not(mask_left_major_img, mask_inv_3);
        cv::bitwise_not(mask_right_major_img, mask_inv_4);

        // 确保掩码和图像的类型匹配
        if (left_inpaint_small_resized.type() != CV_8UC3) {
            left_inpaint_small_resized.convertTo(left_inpaint_small_resized, CV_8UC3);
        }
        if (right_inpaint_small_resized.type() != CV_8UC3) {
            right_inpaint_small_resized.convertTo(right_inpaint_small_resized, CV_8UC3);
        }
        if (left_inpaint_major_resized.type() != CV_8UC3) {
            left_inpaint_major_resized.convertTo(left_inpaint_major_resized, CV_8UC3);
        }
        if (right_inpaint_major_resized.type() != CV_8UC3) {
            right_inpaint_major_resized.convertTo(right_inpaint_major_resized, CV_8UC3);
        }
        if (mask_inv_1.type() != CV_8UC1) {
            mask_inv_1.convertTo(mask_inv_1, CV_8UC1);
        }
        if (mask_inv_2.type() != CV_8UC1) {
            mask_inv_2.convertTo(mask_inv_2, CV_8UC1);
        }
        if (mask_inv_3.type() != CV_8UC1) {
            mask_inv_3.convertTo(mask_inv_3, CV_8UC1);
        }
        if (mask_inv_4.type() != CV_8UC1) {
            mask_inv_4.convertTo(mask_inv_4, CV_8UC1);
        }

        // 使用掩码将 `inpaint_img_resized` 的颜色应用到白色区域
        // 将第一步中的mask区域内白色像素保留
        cv::Mat inpaint_part_1, inpaint_part_2, inpaint_part_3, inpaint_part_4;
        cv::bitwise_and(left_inpaint_small_resized, left_inpaint_small_resized, inpaint_part_1, mask_left_img);
        cv::bitwise_and(right_inpaint_small_resized, right_inpaint_small_resized, inpaint_part_2, mask_right_img);
        cv::bitwise_and(left_inpaint_major_resized, left_inpaint_major_resized, inpaint_part_3, mask_left_major_img);
        cv::bitwise_and(right_inpaint_major_resized, right_inpaint_major_resized, inpaint_part_4, mask_right_major_img);
        // 使用反向掩码将 `left_img` 保留黑色区域的原始颜色
        // 将原图中mask区域外黑色像素保留
        cv::Mat left_part, right_part;
        cv::bitwise_and(left_img, left_img, left_part, mask_inv_1);
        cv::bitwise_and(right_img, right_img, right_part, mask_inv_2);
        // 合并两个图像部分
        // 将mask黑白区域进行相加得到最终结果
        cv::Mat blended_img_1, blended_img_2;
        cv::add(left_part, inpaint_part_1, blended_img_1);
        cv::add(right_part, inpaint_part_2, blended_img_2);

        // 将主要区域的mask区域进行复制
        left_inpaint_major_resized.copyTo(blended_img_1, mask_left_major_img);
        right_inpaint_major_resized.copyTo(blended_img_2, mask_right_major_img);

        // 去除白色噪点
        cv::Mat tmpLeftMsk, tmpRightMsk;
        cv::inRange(blended_img_1, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), tmpLeftMsk);
        cv::inRange(blended_img_2, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), tmpRightMsk);
        cv::inpaint(blended_img_1, tmpLeftMsk, blended_img_1, 7, cv::INPAINT_TELEA);
        cv::inpaint(blended_img_2, tmpRightMsk, blended_img_2, 7, cv::INPAINT_TELEA);

        // 保存最终图像
        cv::imwrite(savepict + "left_final.png", blended_img_1);
        cv::imwrite(savepict + "right_final.png", blended_img_2);
    }
}

int main() {


    // std::string orig_image = argv[1]; //original image input
    // std::string disp_image = argv[2]; //depth input
    // std::string orig_image_inpaint = argv[3]; //original image after inpainting
    // std::string disp_image_inpaint = argv[4]; //depth image after inpainting
    // std::string save_path = argv[5]; //path save
    // std::string img_id = argv[6]; //image name

    std::string orig_image         = "/home/rei/Documents/Project/gen_left_right/input/1_depth.png";
    std::string disp_image         = "/home/rei/Documents/Project/gen_left_right/input/01_depth.png";
    std::string orig_image_inpaint = "/home/rei/Documents/Project/gen_left_right/output_1/iopaint_lama/left_lama.png";
    std::string disp_image_inpaint = "/home/rei/Documents/Project/gen_left_right/output_1/iopaint_lama/right_lama.png";
    std::string save_path          = "/home/rei/Documents/Project/gen_left_right/path";
    std::string img_id             = "1";
    if (save_path.back() != '/') {
        save_path += '/';
    }
    int step = 2;

    main_process(orig_image, disp_image, orig_image_inpaint, disp_image_inpaint, save_path, img_id, step);
    return 0;
}
