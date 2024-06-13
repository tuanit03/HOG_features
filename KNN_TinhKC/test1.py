import numpy as np
import cv2
from matplotlib import pyplot as plt

np.set_printoptions(precision=2, suppress=True)
epsilon = 1e-6

imagePath = 'anhCho.jpg'
image = cv2.imread(imagePath)
image = cv2.resize(image, (128, 128))


np_im2 = image.copy()
gx = cv2.Sobel(np_im2, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(np_im2, cv2.CV_32F, 0, 1, ksize=1)

mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
mag = mag.astype(int)
angle = angle.astype(int)
print(mag.shape)
print(angle.shape)

# plt.figure(figsize=(12, 10))
# plt.imshow(mag)
# plt.figure(figsize=(12, 10))
# plt.imshow(angle)
# plt.show()


maxChan = np.argmax(mag, axis=2)
maxmag = np.zeros(maxChan.shape)
for r in range(maxChan.shape[0]):
    for c in range(maxChan.shape[1]):
        maxmag[r, c] = mag[r, c, maxChan[r, c]]
print(maxmag.shape)
# plt.figure(figsize=(12, 10))
# plt.imshow(maxmag)
# plt.show()

maxangle = np.zeros(maxChan.shape)
for r in range(maxChan.shape[0]):
    for c in range(maxChan.shape[1]):
        maxangle[r, c] = angle[r, c, maxChan[r, c]]
print(maxangle.shape)


def anglemapper(x):
    if x >= 180:
        return x - 180
    else:
        return x


def createHist(AngArray, MagArray, BS=20, BINS=9):
    hist = np.zeros(BINS)
    for r in range(AngArray.shape[0]):
        for c in range(AngArray.shape[1]):
            # print(AngArray[r,c])
            binel, rem = np.divmod(AngArray[r, c], BS)
            weightR = rem * 1.0 / BS
            weightL = 1 - weightR
            deltaR = MagArray[r, c] * weightR
            deltaL = MagArray[r, c] * weightL
            binL = int(binel)
            binR = np.mod(binL + 1, BINS)
            hist[binL] += deltaL
            hist[binR] += deltaR
    return hist


vfunc = np.vectorize(anglemapper)
mappedAngles = (vfunc(maxangle))

spotHist = createHist(mappedAngles, maxmag)
print(spotHist)
# plt.bar(range(9), spotHist)
# plt.xticks(range(9), [0, 20, 40, 60, 80, 100, 120, 140, 160])
# plt.title("Histogram of Gradients in selected patch")
# plt.show()


# Kích thước của ảnh
H, W = image.shape[:2]

# Kích thước của một cell và một block
cell_size = 8
block_size = 2

# Số lượng cell trên mỗi hàng và cột
num_cells_y = ((H - (cell_size * block_size)) // cell_size) + 1
num_cells_x = ((W - (cell_size * block_size)) // cell_size) + 1

# Khởi tạo histogram rỗng cho toàn bộ ảnh
hist_image = []

# Lặp qua từng block trên ảnh
for y in range(0, num_cells_y):
    for x in range(0, num_cells_x):
        # Khởi tạo histogram rỗng cho block hiện tại
        hist_block = []

        # Lặp qua từng cell trong block
        for j in range(block_size):
            for i in range(block_size):
                # Tính toán vị trí của cell hiện tại
                cell_y = y + j
                cell_x = x + i

                # Tính toán vị trí pixel tương ứng trên ảnh
                pixel_y = cell_y * cell_size
                pixel_x = cell_x * cell_size

                # Tính histogram cho cell hiện tại và thêm vào histogram của block
                spotAngles = mappedAngles[pixel_y:pixel_y + cell_size, pixel_x:pixel_x + cell_size]
                spotMag = maxmag[pixel_y:pixel_y + cell_size, pixel_x:pixel_x + cell_size]
                hist_cell = createHist(spotAngles, spotMag)
                hist_block.extend(hist_cell)

        # Chuẩn hóa histogram của block và thêm vào histogram của ảnh
        l2norm = np.sqrt(np.sum(np.array(hist_block) ** 2))
        hist_block_normed = hist_block / (l2norm + epsilon)
        hist_image.extend(hist_block_normed)

# Bây giờ, hist_image chứa histogram chuẩn hóa của toàn bộ ảnh
print(len(hist_image))
