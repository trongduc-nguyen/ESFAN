import torch
import argparse
import importlib

def check_checkpoint(args):
    """
    Hàm để load checkpoint, tạo ảnh giả và kiểm tra shape của output (CAM).
    """
    print("="*50)
    print("Bắt đầu kiểm tra checkpoint...")
    
    # --- 1. Định nghĩa kiến trúc model ---
    # Sử dụng model Net_CAM để lấy đầu ra là bản đồ kích hoạt (CAM)
    # Import model từ file baseline chúng ta đã tạo
    try:
        # Sử dụng getattr để import động dựa trên argument
        model_module = importlib.import_module(args.network)
        model = getattr(model_module, 'Net_CAM')(n_class=args.n_class)
        print(f"Đã khởi tạo thành công model 'Net_CAM' từ '{args.network}'")
        print(f"Số lớp (n_class) được cấu hình: {args.n_class}")
    except Exception as e:
        print(f"Lỗi khi khởi tạo model: {e}")
        return

    # --- 2. Load checkpoint ---
    try:
        checkpoint_path = args.checkpoint_path
        print(f"Đang tải checkpoint từ: {checkpoint_path}")
        
        # Load state_dict vào CPU để tránh lỗi nếu máy không có GPU
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Lớp Net_CAM của chúng ta có một model con tên là 'model'
        # Do đó, cần load state_dict vào model.model
        model.load_state_dict(state_dict)
        
        print("Tải checkpoint thành công!")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file checkpoint tại '{checkpoint_path}'")
        return
    except Exception as e:
        print(f"Lỗi khi tải checkpoint: {e}")
        # In ra các key trong checkpoint để debug nếu có lỗi key mismatch
        print("\nCác key có trong checkpoint:")
        for key in state_dict.keys():
            print(key)
        return
        
    # Chuyển model sang chế độ eval
    model.eval()

    # --- 3. Tạo ảnh giả ---
    # batch_size = 1, 3 kênh màu, kích thước 224x224
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)
    print(f"\nĐã tạo ảnh giả (dummy input) với shape: {dummy_input.shape}")

    # --- 4. Forward qua model ---
    print("Thực hiện forward pass...")
    with torch.no_grad(): # Không cần tính gradient khi kiểm tra
        output_cam = model(dummy_input)
    print("Forward pass hoàn tất.")

    # --- 5. In ra shape của output ---
    print("\n--- KẾT QUẢ KIỂM TRA ---")
    print(f"Shape của output (CAM): {output_cam.shape}")
    
    # Kiểm tra các chiều
    is_shape_correct = True
    if output_cam.shape[0] != 1:
        print(f"Lỗi: Batch size không phải là 1 (kỳ vọng: 1, thực tế: {output_cam.shape[0]})")
        is_shape_correct = False
        
    if output_cam.shape[1] != args.n_class:
        print(f"Lỗi: Số kênh/lớp không khớp (kỳ vọng: {args.n_class}, thực tế: {output_cam.shape[1]})")
        is_shape_correct = False
        
    if is_shape_correct:
        print("\n=> Chúc mừng! Shape của output HOÀN TOÀN CHÍNH XÁC.")
        print(f"   - Batch size: 1")
        print(f"   - Số lớp: {output_cam.shape[1]} (đúng bằng n_class)")
        print(f"   - Kích thước CAM: {output_cam.shape[2]}x{output_cam.shape[3]}")
    else:
        print("\n=> Lỗi: Shape của output KHÔNG chính xác. Vui lòng kiểm tra lại kiến trúc model hoặc checkpoint.")
        
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kiểm tra checkpoint của model baseline")
    
    # Argument cho đường dẫn đến checkpoint
    parser.add_argument("--checkpoint_path", 
                        default='checkpoints/stage1_baseline_luad.pth', 
                        type=str,
                        help="Đường dẫn đến file checkpoint .pth đã được huấn luyện.")
    
    # Argument cho file định nghĩa model
    parser.add_argument("--network", 
                        default="network.resnet38_cls_baseline", 
                        type=str,
                        help="Module python định nghĩa kiến trúc model (ví dụ: network.resnet38_cls_baseline).")
    
    # Argument cho số lớp
    parser.add_argument("--n_class", 
                        default=4, 
                        type=int,
                        help="Số lượng lớp của bộ dữ liệu.")

    # Argument cho kích thước ảnh
    parser.add_argument("--img_size",
                        default=224,
                        type=int,
                        help="Kích thước của ảnh đầu vào (ví dụ: 224 cho 224x224).")

    args = parser.parse_args()
    
    check_checkpoint(args)