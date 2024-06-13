import cv2
from matplotlib import pyplot as plt


# Hàm tiền xử lý ảnh: đọc ảnh từ đường dẫn và thay đổi kích thước ảnh thành 128x128
def TienXuLy(imagePath):
    image = cv2.imread(imagePath)  # Đọc ảnh từ đường dẫn
    image = cv2.resize(image, (128, 128))  # Thay đổi kích thước ảnh
    return image


# Hàm vẽ block: vẽ một block gồm 4 cell (8x8 pixel) trên ảnh
def VeBlock(np_im):
    Location = [50, 70]  # Vị trí bắt đầu của block
    PatchSize = [16, 16]  # Kích thước của block
    numlinesY = int(PatchSize[0] / 8)  # Số lượng cell theo chiều dọc
    numlinesX = int(PatchSize[1] / 8)  # Số lượng cell theo chiều ngang

    plt.figure(figsize=(12, 10))  # Tạo một khung hình mới
    # Vẽ một hình chữ nhật xung quanh block
    cv2.rectangle(np_im, (Location[1], Location[0]),
                  (Location[1] + PatchSize[1], Location[0] + PatchSize[0]),
                  (0, 0, 255), 1)

    # Vẽ các đường kẻ để chia block thành các cell
    for x in range(numlinesX):
        cv2.line(np_im, (Location[1] + 8 * (x + 1), Location[0]),
                 (Location[1] + 8 * (x + 1), Location[0] + PatchSize[0]),
                 (0, 0, 255), 1)
    for y in range(numlinesY):
        cv2.line(np_im, (Location[1], Location[0] + 8 * (y + 1)),
                 (Location[1] + PatchSize[1], Location[0] + 8 * (y + 1)),
                 (0, 0, 255), 1)
    plt.imshow(np_im)  # Hiển thị ảnh
    plt.show()  # Hiển thị khung hình


# Hàm chính
if __name__ == '__main__':
    image = TienXuLy('anhCho.jpg')  # Tiền xử lý ảnh
    VeBlock(image)  # Vẽ block trên ảnh
