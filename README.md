# Pronunciation-Dataset-220K — Nền Tảng Dữ Liệu Quy Mô Lớn Cho Chẩn Đoán Lỗi Phát Âm Tiếng Anh L2

Dự án chuyên biệt độc lập nhằm xây dựng, chuẩn hóa và kiểm định **220.000 bản thu âm có nhãn từ 2.200 người nói độc lập** cho bài toán **Phát hiện và Chẩn đoán Lỗi Âm vị Tiếng Anh L2 (Mispronunciation Detection and Diagnosis - MDD)** trên 3 loại lỗi chính: **Substitution (S), Omission (O), Insertion (I)** theo chuẩn topo Exact-Gap ($N$ targets / $N+1$ gaps).

Dự án được khởi tạo độc lập tách rời khỏi `New-Method-Research` nhằm đảm bảo tính toàn vẹn, bảo vệ các tiến trình mô hình SM13/CROTTC hiện hữu và ngăn chặn rò rỉ dữ liệu.

---

## 1. Cấu Trúc Phân Bổ Quy Mô 220.000 Bản Thu / 2.200 Người Nói

| Phân Vùng Dữ Liệu | Số Người Nói Độc Lập | Số Bản Thu (Utterances) | Thời Lượng Danh Định (~4,5s/câu) | Bản Chất Dữ Liệu & Gán Nhãn | Vai Trò Kỹ Thuật |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Tập Huấn Luyện (Train)** | **1.400** | **140.000** | ~175,0 giờ | Synthetic S/O/I Perturbation (LibriTTS-R 24kHz) + Core VN | Huấn luyện mô hình quy mô lớn, zero chi phí gán nhãn tay |
| **Tập Phát Triển (Dev)** | **200** | **20.000** | ~25,0 giờ | 100% Người Việt thật, Double-blind Annotation | Chọn mô hình (`DEV_FIT` 15k) và Chuẩn hóa calibration (`DEV_CAL` 5k) |
| **Tập Đánh Giá Nội Bộ (Test Int)** | **400** | **40.000** | ~50,0 giờ | 100% Người Việt thật, Double-blind Annotation | Track 1 Độc lập Prompt (30k) + Track 2 Parallel Benchmark (10k) |
| **Tập Đánh Giá Ngoài Miền (Test OOD)**| **200** | **20.000** | ~25,0 giờ | 100% Dữ liệu thực tế / Nhiễu / Trẻ em | Đánh giá độ bền bỉ thiết bị di động (10k) + Thử thách trẻ em (10k) |
| **TỔNG VŨ TRỤ DỮ LIỆU** | **2.200** | **220.000** | **~275,0 giờ** | **100% Bảo Toàn Số Học Tuyệt Đối** | **Mục tiêu R&D quy mô lớn** |

---

## 2. Chiến Lược Dữ Liệu Lai 3 Trụ Cột (Master 3-Pillar Hybrid Strategy)

1. **Trụ cột 1 — Dữ liệu Đánh giá Gold 100% Người Việt (80.000 câu / 800 người nói):**
   - Dành riêng cho `Dev` (20.000 câu) và `Test` (60.000 câu gồm 40k Nội bộ và 20k OOD).
   - Tuyển dụng 800 người nói Việt Nam (40% Bắc / 40% Nam / 20% Trung; 50% Nam / 50% Nữ; CEFR A1-C1).
   - 100% thẩm âm kép độc lập (Double-blind annotation) bởi 2 chuyên gia ngữ âm + Hội chẩn bất đồng cấp cao.
2. **Trụ cột 2 — Sinh lỗi nhân tạo cho Huấn luyện (140.000 câu Synthetic S/O/I):**
   - Tận dụng 585 giờ đọc chuẩn 24kHz sạch của LibriTTS-R (CC BY 4.0).
   - Ứng dụng **Synthetic Perturbation Engine** mô phỏng chính xác các quy luật lỗi chuyển di L1 Việt Nam $\to$ L2 Anh (nuốt phụ âm cuối `/t, s, z, d/`, thế âm xát răng `/θ, ð/`, chèn âm đệm schwa `[ə]`).
   - Đảm bảo tọa độ nhãn S/O/I chính xác 100% tuyệt đối về mặt toán học với chi phí gán nhãn bằng 0$.
3. **Trụ cột 3 — Tích hợp các bộ benchmark mở đã kiểm định:**
   - Nạp **MRPL2** (4.500 câu, CC BY 4.0), **L2-ARCTIC** (3.599 câu, CC BY-NC 4.0), và **SpeechOcean762** (5.000 câu, CC BY 4.0) làm tập đối chuẩn khoa học với thế giới.

---

## 3. Cấu Trúc Thư Mục Dự Án

```
Pronunciation-Dataset-220K/
├── README.md                      # Tài liệu tổng quan dự án
├── GOAL.md                        # Đặc tả mục tiêu và ranh giới ủy quyền
├── docs/                          # Toàn bộ 14 hồ sơ nghiên cứu, kiểm toán và thẩm định
│   ├── MASTER_DATASET_RD_REPORT.md      # Báo cáo tổng thể R&D 220k (11 chương)
│   ├── MASTER_GLOBAL_SEARCH_CATALOG.md  # Toàn thư kiểm kê dữ liệu tiếng nói toàn cầu
│   ├── AUDIT_SOI.md                     # Đặc tả topo exact-gap S/O/I
│   ├── AUDIT_QUALITY.md                 # Quy trình độ tin cậy nhãn & Kappa/Alpha
│   ├── AUDIT_LEAKAGE.md                 # 5 nguyên tắc vàng chống rò rỉ & UUIDv5
│   ├── AUDIT_LEGAL.md                   # Pháp lý bản quyền, BIPA & Nghị định 13/2023
│   ├── QC_AUDIO.md                      # Đặc tả kỹ thuật âm học & Pydantic schema
│   ├── PLAN_PILOT.md                    # Thiết kế Phase 1 Pilot 120 spk / 12k câu
│   ├── PLAN_COST.md                     # Mô hình chi phí thực tế & RTM 33x-42x
│   ├── VERIFY_SOURCES.md                # Hồ sơ thẩm định nguồn độc lập
│   └── VERIFY_CONTRACTS.md              # Hồ sơ kiểm toán hợp đồng & Section 8
├── pipelines/                     # Mã nguồn các đường ống xử lý kỹ thuật
│   ├── synthetic_engine/          # Bộ sinh lỗi âm vị nhân tạo (LibriTTS-R -> Synthetic S/O/I)
│   ├── audio_qc/                  # Module kiểm tra chất lượng âm thanh 16kHz, SNR, Clipping
│   ├── manifest_validator/        # Pydantic v2 data contract & SHA-256 validator
│   └── split_manager/             # Bộ sinh phân vùng chống rò rỉ người nói & prompt
├── data/                          # Thư mục lưu trữ dữ liệu (đã cấu hình .gitignore)
│   ├── raw/                       # Audio thô tải về (bị gitignore)
│   ├── interim/                   # Audio 16kHz mono đã chuẩn hóa
│   ├── processed/                 # Audio kèm nhãn TextGrid / JSON S/O/I hoàn thiện
│   └── manifests/                 # File manifest JSONL chứa SHA-256 metadata
└── tests/                         # Bộ kiểm thử đơn vị & kiểm thử hợp đồng dữ liệu
```

---

## 4. Ranh Giới Ủy Quyền & Tuân Thủ Section 8

Dự án tuân thủ nghiêm ngặt 5 quy tắc cứng:
1. **Zero Unauthorized Spend:** Không tự ý giải ngân hoặc cam kết tài chính khi chưa có phê chuẩn từ người dùng.
2. **Zero Cloud API Uploads:** 100% âm thanh và vector đặc trưng người nói được xử lý cục bộ on-premises, cấm đưa lên Whisper API hay Google Speech API.
3. **Sealed Benchmarks Untouched:** Giữ nguyên vẹn các tập benchmark niêm phong lịch sử (`DEV_CAL`, `held` 3 người của SM13).
4. **Git Repository Hygiene:** Không bao giờ commit file âm thanh thô vào Git.
5. **Baseline Preservation:** Giữ SM13 LoRA làm mốc so sánh khoa học đối chứng.
