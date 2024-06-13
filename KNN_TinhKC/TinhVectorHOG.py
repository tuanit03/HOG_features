import ChuanHoaBlock
import TienXuLy
import TinhGradient
import TinhVectorBlock


def VectorHOG(image, mappedAngles, maxmag):
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
                    hist_cell = TinhVectorBlock.createHist(spotAngles, spotMag)
                    hist_block.extend(hist_cell)

            # Chuẩn hóa histogram của block và thêm vào histogram của ảnh
            hist_block_normed = ChuanHoaBlock.ChuanHoaL2Norm(hist_block)
            hist_image.extend(hist_block_normed)
    return hist_image


def main(image):
    maxmag, maxangle = TinhGradient.TinhGradient(image)
    mappedAngles = TinhVectorBlock.anhXaGoc(maxangle)
    hist_image = VectorHOG(image, mappedAngles, maxmag)
    return hist_image


if __name__ == '__main__':
    image = TienXuLy.TienXuLy('anhCho.jpg')
    maxmag, maxangle = TinhGradient.TinhGradient(image)
    mappedAngles = TinhVectorBlock.anhXaGoc(maxangle)
    hist_image = VectorHOG(image, mappedAngles, maxmag)
    print(len(hist_image))
