import numpy as np
from matplotlib import pyplot as plt
import TienXuLy
import TinhGradient


# Hàm để ánh xạ góc từ 0-360 về 0-180
def anglemapper(x):
    if x >= 180:
        return x - 180
    else:
        return x


# Vector hóa hàm và ánh xạ góc về 0-180
def anhXaGoc(maxangle):
    vfunc = np.vectorize(anglemapper)  # Vector hóa hàm để áp dụng cho mảng numpy
    mappedAngles = (vfunc(maxangle))  # Ánh xạ góc về 0-180
    return mappedAngles


# Hàm để tạo histogram từ mảng góc và mảng độ lớn
def createHist(AngArray, MagArray, BS=20, BINS=9):
    hist = np.zeros(BINS)  # Khởi tạo histogram với số lượng bin
    for r in range(AngArray.shape[0]):
        for c in range(AngArray.shape[1]):
            binel, rem = np.divmod(AngArray[r, c], BS)  # Chia bin
            weightR = rem * 1.0 / BS  # Tính trọng số cho bin bên phải
            weightL = 1 - weightR  # Tính trọng số cho bin bên trái
            deltaR = MagArray[r, c] * weightR  # Tính đóng góp cho bin bên phải
            deltaL = MagArray[r, c] * weightL  # Tính đóng góp cho bin bên trái
            binL = int(binel)
            binR = np.mod(binL + 1, BINS)
            hist[binL] += deltaL  # Cập nhật bin bên trái
            hist[binR] += deltaR  # Cập nhật bin bên phải
    return hist


# Hàm để vẽ histogram
def VeHistBlock(spotHist):
    plt.bar(range(9), spotHist)  # Vẽ histogram
    plt.xticks(range(9), [0, 20, 40, 60, 80, 100, 120, 140, 160])  # Đặt các nhãn cho trục x
    plt.title("Histogram of Gradients in selected patch")  # Đặt tiêu đề cho biểu đồ
    plt.show()  # Hiển thị biểu đồ


# Hàm chính
if __name__ == '__main__':
    image = TienXuLy.TienXuLy('anhCho.jpg')  # Tiền xử lý ảnh
    maxmag, maxangle = TinhGradient.TinhGradient(image)  # Tính gradient theo độ lớn và gradient theo phương
    mappedAngles = anhXaGoc(maxangle)  # Chia bin và ánh xạ góc về 0-180
    spotHist = createHist(mappedAngles, maxmag)  # Tạo histogram từ mảng góc và mảng độ lớn
    print(spotHist)  # In histogram
    VeHistBlock(spotHist)  # Vẽ histogram
