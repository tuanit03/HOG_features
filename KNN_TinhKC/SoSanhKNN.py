import numpy as np
import matplotlib.pyplot as plt
import os
import TinhVectorHOG
import TienXuLy


# Hàm tính khoảng cách Manhattan giữa hai vector HOG
def calculate_manhattan_distance(HOG1, HOG2):
    distance = np.sum(np.abs(HOG1 - HOG2))  # Tính tổng giá trị tuyệt đối của sự khác biệt giữa hai vector
    return distance


# Hàm tính khoảng cách Cosine giữa hai vector HOG
def calculate_cosine_distance(HOG1, HOG2):
    dot_product = np.dot(HOG1, HOG2)  # Tính tích vô hướng của hai vector
    norm_HOG1 = np.linalg.norm(HOG1)  # Tính chuẩn của vector HOG1
    norm_HOG2 = np.linalg.norm(HOG2)  # Tính chuẩn của vector HOG2
    distance = 1 - dot_product / (norm_HOG1 * norm_HOG2)  # Tính khoảng cách Cosine
    return distance


# Hàm tính khoảng cách Euclidean giữa hai vector HOG
def calculate_euclidean_distance(HOG1, HOG2):
    distance = np.sqrt(np.sum((HOG1 - HOG2) ** 2))  # Tính căn bậc hai của tổng bình phương sự khác biệt giữa hai vector
    return distance


# Hàm vẽ biểu đồ khoảng cách Euclidean, Manhattan và Cosine
def plot_distances(distances_euclidean, distances_manhattan, distances_cosine):
    labels_euclidean, values_euclidean = zip(
        *distances_euclidean)  # Chia danh sách distances_euclidean thành hai danh sách riêng biệt
    labels_manhattan, values_manhattan = zip(
        *distances_manhattan)  # Chia danh sách distances_manhattan thành hai danh sách riêng biệt
    labels_cosine, values_cosine = zip(
        *distances_cosine)  # Chia danh sách distances_cosine thành hai danh sách riêng biệt

    fig, axs = plt.subplots(1, 3, figsize=(16, 9))  # Tạo một khung hình mới với 3 subplot

    axs[0].bar(labels_euclidean, values_euclidean, color='b')  # Vẽ biểu đồ cột cho khoảng cách Euclidean
    axs[0].set_title('Euclidean Distances')  # Đặt tiêu đề cho subplot
    axs[0].set_ylabel('Distance')  # Đặt nhãn cho trục y

    axs[1].bar(labels_manhattan, values_manhattan, color='r')  # Vẽ biểu đồ cột cho khoảng cách Manhattan
    axs[1].set_title('Manhattan Distances')  # Đặt tiêu đề cho subplot
    axs[1].set_ylabel('Distance')  # Đặt nhãn cho trục y

    axs[2].bar(labels_cosine, values_cosine, color='g')  # Vẽ biểu đồ cột cho khoảng cách Cosine
    axs[2].set_title('Cosine Distances')  # Đặt tiêu đề cho subplot
    axs[2].set_ylabel('Distance')  # Đặt nhãn cho trục y

    plt.tight_layout()  # Điều chỉnh layout để không bị chồng chéo
    plt.show()  # Hiển thị khung hình


# Hàm phân loại ảnh dựa trên đặc trưng HOG
def classify_image(new_image, dataset, K):
    new_image_HOG = np.array(TinhVectorHOG.main(new_image))  # Tính đặc trưng HOG cho ảnh mới

    distances_euclidean = []  # Khởi tạo danh sách để lưu khoảng cách Euclidean
    distances_manhattan = []  # Khởi tạo danh sách để lưu khoảng cách Manhattan
    distances_cosine = []  # Khởi tạo danh sách để lưu khoảng cách Cosine

    for image_name, image in dataset.items():  # Lặp qua từng ảnh trong tập dữ liệu
        image_HOG = np.array(TinhVectorHOG.main(image))  # Tính đặc trưng HOG cho ảnh

        distances_euclidean.append((image_name, calculate_euclidean_distance(new_image_HOG,
                                                                             image_HOG)))  # Tính và thêm khoảng cách Euclidean vào danh sách
        distances_manhattan.append((image_name, calculate_manhattan_distance(new_image_HOG,
                                                                             image_HOG)))  # Tính và thêm khoảng cách Manhattan vào danh sách
        distances_cosine.append((image_name, calculate_cosine_distance(new_image_HOG,
                                                                       image_HOG)))  # Tính và thêm khoảng cách Cosine vào danh sách

    distances_euclidean.sort(key=lambda x: x[1])  # Sắp xếp danh sách khoảng cách Euclidean theo thứ tự tăng dần
    neighbors_distances_euclidean = distances_euclidean[:K]  # Lấy K khoảng cách Euclidean nhỏ nhất

    distances_manhattan.sort(key=lambda x: x[1])  # Sắp xếp danh sách khoảng cách Manhattan theo thứ tự tăng dần
    neighbors_distances_manhattan = distances_manhattan[:K]  # Lấy K khoảng cách Manhattan nhỏ nhất

    distances_cosine.sort(key=lambda x: x[1])  # Sắp xếp danh sách khoảng cách Cosine theo thứ tự tăng dần
    neighbors_distances_cosine = distances_cosine[:K]  # Lấy K khoảng cách Cosine nhỏ nhất

    return neighbors_distances_euclidean, neighbors_distances_manhattan, neighbors_distances_cosine  # Trả về K khoảng cách nhỏ nhất theo từng độ đo


# Hàm tải ảnh từ thư mục
def load_images_from_folder(folder):
    images = {}  # Khởi tạo từ điển để lưu ảnh
    for filename in os.listdir(folder):  # Lặp qua từng tệp trong thư mục
        img = TienXuLy.TienXuLy(os.path.join(folder, filename))  # Đọc và tiền xử lý ảnh
        if img is not None:  # Nếu ảnh không rỗng
            images[filename] = img  # Thêm ảnh vào từ điển
    return images  # Trả về từ điển chứa ảnh


# Hàm chính
if __name__ == '__main__':
    folder = "dataset"  # Đường dẫn đến thư mục chứa tập dữ liệu
    dataset = load_images_from_folder(folder)  # Tải ảnh từ thư mục

    image1 = TienXuLy.TienXuLy('anhCho.jpg')  # Tiền xử lý ảnh

    distances_euclidean, distances_manhattan, distances_cosine = classify_image(image1, dataset, 3)  # Phân loại ảnh

    plot_distances(distances_euclidean, distances_manhattan, distances_cosine)  # Vẽ biểu đồ khoảng cách
