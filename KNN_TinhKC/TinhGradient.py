import numpy as np
import cv2
from matplotlib import pyplot as plt
import TienXuLy


# Hàm vẽ gradient theo độ lớn và gradient theo phương
def VeGradient(maxmag, maxangle):
    plt.figure(figsize=(10, 5))  # Tạo một khung hình mới

    plt.subplot(1, 2, 1)  # Tạo một subplot trong khung hình (1 hàng, 2 cột, vị trí thứ nhất)
    plt.imshow(maxmag)  # Vẽ gradient theo độ lớn
    plt.title('Gradient theo độ lớn')  # Đặt tiêu đề cho ảnh

    plt.subplot(1, 2, 2)  # Tạo một subplot khác trong cùng khung hình (1 hàng, 2 cột, vị trí thứ hai)
    plt.imshow(maxangle)  # Vẽ gradient theo phương
    plt.title('Gradient theo phương')  # Đặt tiêu đề cho ảnh

    plt.show()  # Hiển thị khung hình


# Hàm tính gradient theo độ lớn và gradient theo phương
def TinhGradient(np_im2):
    gx = cv2.Sobel(np_im2, cv2.CV_32F, 1, 0, ksize=1)  # Tính gradient theo hướng x
    gy = cv2.Sobel(np_im2, cv2.CV_32F, 0, 1, ksize=1)  # Tính gradient theo hướng y

    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)  # Chuyển gradient từ dạng Cartesian sang dạng cực
    mag = mag.astype(int)  # Chuyển đổi kiểu dữ liệu của gradient theo độ lớn thành số nguyên
    angle = angle.astype(int)  # Chuyển đổi kiểu dữ liệu của gradient theo phương thành số nguyên

    maxChan = np.argmax(mag, axis=2)  # Tìm chỉ số của giá trị lớn nhất trong mỗi kênh màu
    maxmag = np.zeros(maxChan.shape)  # Khởi tạo mảng để lưu gradient theo độ lớn lớn nhất
    for r in range(maxChan.shape[0]):
        for c in range(maxChan.shape[1]):
            maxmag[r, c] = mag[r, c, maxChan[r, c]]  # Lấy gradient theo độ lớn lớn nhất từng pixel

    maxangle = np.zeros(
        maxChan.shape)  # Khởi tạo mảng để lưu gradient theo phương tương ứng với gradient theo độ lớn lớn nhất
    for r in range(maxChan.shape[0]):
        for c in range(maxChan.shape[1]):
            maxangle[r, c] = angle[
                r, c, maxChan[r, c]]  # Lấy gradient theo phương tương ứng với gradient theo độ lớn lớn nhất

    return maxmag, maxangle  # Trả về gradient theo độ lớn và gradient theo phương lớn nhất


# Hàm chính
if __name__ == '__main__':
    image = TienXuLy.TienXuLy('anhCho.jpg')  # Tiền xử lý ảnh
    maxmag, maxangle = TinhGradient(image)  # Tính gradient theo độ lớn và gradient theo phương
    VeGradient(maxmag, maxangle)  # Vẽ gradient theo độ lớn và gradient theo phương
