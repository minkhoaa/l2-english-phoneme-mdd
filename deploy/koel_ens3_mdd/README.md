# Koel + 3-Seed `SingleFactorV2` MDD Inference Bundle (`S0_clean_ens3_o5_o4`)

**Trạng thái:** `MAJOR_RESEARCH_PILOT_MILESTONE` (chưa duyệt triển khai đại trà; chờ xác nhận câu ngắn người Việt độc lập `EXP-6` và lớp G2P đa biến thể phát âm).

## 1. Thành phần gói
- **`bundle_s0_clean_ens3_o5_o4.pt`** (`15,94 MiB`, SHA-256 `44a978287b5d2e301688ee9a7f1cb9c222a93a2d371a607955ab8ff8b62042cf`): chứa đủ 3 head `SingleFactorV2 (iso_no_hwin)` (`seed 20261121..23`, train trên split sạch `clean_11` tách tuyệt đối 4 người bảo vệ `test_sealed`), tham số chuẩn hóa Fit, và policy khóa thuần trên Cal (`prob_mean`, `tau_t = 0,35`, `tau_g = 0,38`, Gap Platt `T = 1,191 / b = -2,570`, Cal L1 `0,39/1k`).
- **`engine.py`**: API Python (`KoelEns3MDDEngine`) và CLI chạy 1 lượt forward Base Koel đóng băng + 3 head ensemble từ file `.wav` 16 kHz.
- **`DEPLOY_VERIFICATION.json`** & **`BUNDLE_CARD.json`**: biên nhận kiểm đối khớp từ file `.wav` thô và độ trễ đo thực tế.

## 2. Kết quả kiểm chứng trên `clean_11` (khóa thuần Cal, niêm phong trước chấm)
- **Strict micro-F1 (S/O/I):** Held-6 **`66,24%`** (`tAP 0,7753`, `gAP 0,5622`), EpaDB TEST **`69,02%`** (`tAP 0,8149`), Spontaneous lát Held-6 **`64,83%`** (`gAP 0,5401`), SAA-27 **`43,92%`** (`tAP 0,4709`).
- **Chốt an toàn bản ngữ:** TIMIT Core **`0,562 FA/câu`** (`106` target + `2` gap / 192 câu), TIMIT Corruption Control Recall **`97,13%`**.
- **Đối khớp từ `.wav` thô (15 câu, 997 slot) vs Parquet niêm phong:** Target **`491/491`** (`max |Δ| = 5,5e-7`), Gap **`506/506`** (`max |Δ| = 1,02e-6`).
- **Độ trễ đầu-cuối (`WAV → C/S/O/I`):**
  - GTX 1650 câu ngắn (`50` Held-6): P50 **`155,8 ms`** / P95 **`220,6 ms`** (`1.646 MiB` VRAM).
  - GTX 1650 đoạn Stella dài 22–49s: P50 **`1.477,6 ms`** (`3.498 MiB` VRAM).
  - CPU-only câu ngắn: P50 **`1.791,1 ms`**.

## 3. Cách sử dụng nhanh

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m deploy.koel_ens3_mdd.engine \
  --wav path/to/sample_16k.wav \
  --phones "HH AH L OW" \
  --device cuda
```

```python
from deploy.koel_ens3_mdd import KoelEns3MDDEngine

engine = KoelEns3MDDEngine(device="cuda")
result = engine.score_wav("path/to/sample_16k.wav", ["HH", "AH", "L", "OW"])
```

## 4. Giới hạn bắt buộc lưu ý trước khi đưa ra người dùng cuối
1. Chưa được xác nhận trên tập câu ngắn người học Việt Nam mới + micro ngoài phòng thu (`EXP-6`).
2. Vẫn còn trung bình `0,56 FA/câu` trên người bản ngữ và yếu hơn trên bài đọc dài (`SAA-27 43,92%`).
3. Cần bổ sung lớp G2P hỗ trợ nhiều cách phát âm chuẩn hợp lệ trước khi kết nối trực tiếp với văn bản thô.
