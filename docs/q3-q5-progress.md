# Nhật ký tích hợp Q3 + Q5

## Bước 1 — Mô hình Q3 (03/11/2026)

Mục tiêu: bổ sung phần kiến trúc và forward của DigitClassificationModel để review độc lập trước khi tích hợp loss và huấn luyện.

Thay đổi:
- Đầu vào là batch ảnh MNIST đã trải phẳng, shape (B, 784).
- Hai lớp ẩn: Linear(784, 256), ReLU, Linear(256, 128), ReLU.
- Lớp cuối Linear(128, 10) trả logits (B, 10).
- Không dùng ReLU hoặc softmax ở đầu ra vì bước loss tiếp theo sẽ dùng cross entropy nhận logits.
- Giữ nguyên import, tên class và chữ ký hàm.

Lý do chọn: mạng fully connected hai lớp ẩn là cấu hình khởi đầu có khả năng học đặc trưng phi tuyến. Chưa kết luận cấu hình tốt hay đạt ngưỡng đề chỉ từ kiểm tra forward; cần huấn luyện và đánh giá ở bước sau.

Kiểm thử đã chạy từ thư mục machinelearning:

```text
python -m unittest discover -s tests -v
Ran 1 test in 0.270s
OK
```

Test gồm ba trường hợp batch size 1, 4, 100; kiểm tra shape logits, giá trị hữu hạn và gradient truyền về input cũng như các tham số của mô hình.

Phạm vi kết quả: xác nhận forward và đồ thị gradient hoạt động. Nhánh này chưa có loss/huấn luyện Q3 nên chưa chạy autograder q3 hoàn chỉnh và chưa có accuracy cho bước này.

Bước kế tiếp: triển khai digitclassifier_loss và kiểm tra shape (1,), giá trị loss, gradient theo nhãn one-hot.
