import numpy as np
epsilon = 1e-6


def ChuanHoaL2Norm(hist_block):
    # Chuẩn hóa histogram của block và thêm vào histogram của ảnh
    l2norm = np.sqrt(np.sum(np.array(hist_block) ** 2))
    hist_block_normed = hist_block / (l2norm + epsilon)
    return hist_block_normed


def ChuanHoaL1Norm(hist_block):
    # Chuẩn hóa histogram của block theo L1-norm
    l1norm = np.sum(np.abs(hist_block))
    hist_block_normed = hist_block / (l1norm + epsilon)
    return hist_block_normed


def ChuanHoaL1Sqrt(hist_block):
    # Chuẩn hóa histogram của block theo L1-sqrt
    l1norm = np.sum(np.abs(hist_block))
    hist_block_normed = np.sqrt(hist_block) / (l1norm + epsilon)
    return hist_block_normed
