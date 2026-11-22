# Giao thức chiến dịch nâng cấp Biểu diễn Âm học & Backbone (`BACKBONE_REP_CAMPAIGN_20261122_v1`)

**Ngày khóa:** 2026-11-22 · **Nền tảng:** GTX 1650 (4 GiB VRAM, trần `≤ 3.400 MiB`), chạy 100% offline sau khi tải snapshot.

## 1. Mục tiêu & giả thuyết khoa học
Kiểm tra xem encoder khác có bổ sung thông tin cho Koel đóng băng, không giả định trước có “trần” CTC. Audit lịch sử của head `fixed` có 63–72% FN thuộc lát Koel tự tin `can_lp_vit_max > -0,2`; CTC thường cho span rất ngắn, nhưng điều đó không chứng minh hidden state đã mất chi tiết âm học. So sánh WavLM-Large/Base+ SSL, HuPER (WavLM-Large đã học nhận dạng âm) và Whisper-Small; kiểm lại giá trị của pooling theo vùng và local cross-attention. Biên CharSiu có lưới 10 ms, không phải biên do người gán nhãn.

## 2. Quy chế split & bảo toàn dữ liệu
- **Split duy nhất được phép (`clean_11`):** `3.681 Fit / 635 Cal / 2.446 prompt Eval`, `11` người L2-ARCTIC Fit + `26` EpaDB + `334` TIMIT. So SHA proposal, roster công khai và WAV/cache. Các tập không trùng UID/WAV xuyên vai; không có 4 người bảo vệ. Không suy ra độc lập ở cấp pretraining, hay so được UID/WAV trực tiếp với tập niêm phong.
- **Eval chẩn đoán, không phải tập mới:** full Spontaneous có `85/109` câu từ người Fit hoặc Cal, nên chỉ dùng lát Held-6 (`24` câu) trong cổng chính. SAA-27 cùng một bài Stella, báo riêng. Không mở `test_sealed.jsonl`, không sửa `GOAL.md` hoặc checkpoint cũ.
- **Chọn chỉ trên Fit/Cal:** niêm phong dự đoán không nhãn, card trọng số và mảng Cal. Stage 6 xác minh toàn bộ seal và khóa policy recall-matched trước khi ghép nhãn lịch sử. Smoke dùng nhãn thủ tục để kiểm wiring; không đọc gold lịch sử, không có giá trị đo chất lượng.

## 3. Các giai đoạn (Stage 0 → 6)
- **Stage 0:** `S0_clean` khởi tạo mới (`iso_no_hwin`, 14 epoch, batch 16, seed `20261121..23`).
- **Stage 1:** 4 encoder đóng băng, Fit/Cal slot-pooling, cùng mẫu Fit tối đa 32k target và 24k gap. `RidgeCV` trên Fit (`50/250/1.000/5.000/20.000`). Qua Cal nếu tAP không giảm quá `0,005`, native target FA tại ~70% Cal recall không tăng quá `0,2 pp`, và đạt một trong hai: (i) tAP `+0,005` và lát mù `+0,01 AP` (≥25 positives); (ii) gAP `+0,02` (≥25 positives). Xếp hạng `0,5·tAP + 0,3·blindAP + 0,2·gAP`. Tối đa 2 encoder qua cổng; không tự đôn nhánh trượt. Thử CharSiu/prototype cho Top-1 bằng Fit/Cal. Prototype từ Fit không phải nhãn vàng mới; A3 là một gói tín hiệu.
- **Stage 2:** `A1–A4` 14 epoch joint + tối đa 8 epoch gap-only; giữ chung loss có giám sát với S0 ở joint. Kiểm SHA nhánh đóng băng và parity target trước/sau pha gap. Không có Top-2 thì bỏ A2/A4; không có Top-1 thì bỏ Stage 2–3.
- **Stage 3:** `B1/B2` local cross-attention trên cửa sổ 17 frame; B2 thêm cửa sổ Koel L15/L18. Đây là các cửa sổ neo đơn điệu theo trật tự CTC, không phải bộ căn chỉnh toàn chuỗi mới. 12 epoch, microbatch 4 × tích lũy 4; inference giới hạn 256 slot đệm, Stella chạy riêng, không cắt WAV ngoài quy tắc 30s của Whisper.
- **Stage 4:** `C1/C2` chạy riêng dù các probe đóng băng trượt. Ưu tiên Large độc lập với hạng encoder đóng băng; chỉ fallback xuống Base+ do VRAM trước C1. C2 và các seed lặp khóa đúng backbone đã dùng. LoRA r8 trên 8/6 tầng cuối, microbatch 1 × tích lũy 16, 8 epoch trên toàn Fit; kiểm gradient projection trên WAV Fit dài nhất. `C2` thêm 2 epoch phone/AF-bundle CTC từ tối đa 2.400 WAV Tier A (TIMIT Fit dư + DoReCo TRAIN), không chia đều frame giả nhãn. Đây là so sánh hai quy trình, không tách được tác động của thêm dữ liệu với thêm update; AF là thuộc tính suy từ nhãn âm, không đo trực tiếp cử động.
- **Stage 5:** tối đa 2 nhánh qua Cal so S0 cùng seed: FA native `+≤2/1k`, tAP không giảm quá `0,005`, và strict F1 `+≥0,5 pp` hoặc tAP/gAP `+≥0,01`. Lặp seed `20261122/23`; seed `20261121` đã dùng chọn mô hình. Nếu teacher nặng qua cổng thì thử `D0` Base+ no-KD vs `D1` Base+ KD cùng seed. Các kiến trúc B/C khác ngân sách tối ưu, không diễn giải như ablation một biến so S0.
- **Stage 6:** `PRE_GOLD_EVAL_LOCK.json` xác minh SHA, khoá key `(cohort, uid, domain, index)` và policy Cal không tách điểm bằng nhau. Đếm độc lập `S/O/I` và control sau ghép; native/recall-match, bootstrap theo người, lát mù và từng người Held-6. Đo phần encoder ở batch 1 có warmup; không báo p95 end-to-end, không đo lại PER.

## 4. Tiêu chí đánh giá chẩn đoán (so với `S0_clean` 3 seed)
- **Guard từng seed, cả hai điểm vận hành:** `Control Recall ≥95%`, số FA TIMIT Core không tăng so S0. Tất cả dùng cùng cache nhận dạng Koel chỉ đọc.
- **Hiệu ứng chẩn đoán:** tăng trung bình `≥0,5 pp` strict F1 hoặc `≥0,01` tAP/gAP trên Held-6 / EpaDB / Spontaneous lát Held-6, cùng chiều 3/3 seed. SAA-27 không tham gia tiêu chí. Không tự thăng checkpoint.

## 5. Chạy và bàn giao

```bash
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -u -m research.BACKBONE_REP_CAMPAIGN_20261122_v1.run_campaign \
  --out research/BACKBONE_REP_CAMPAIGN_20261122_v1/runs/full_clean11_v1 --mode full --split-mode clean_11 --device cuda
```

Dùng `--resume` với cùng đường dẫn, mode, CUDA, mã và hashes; giữ các seed đã xong, giữ phần dở trong `_partial_attempts/`, chạy lại từ đầu seed đang dở. `PROCESS_STATUS.json` cho PID/lỗi hoặc SHA kết quả, `PIPELINE_STATE.json` cho các arm đã xong; `RUN_BINDINGS.json` khóa các snapshot đã tải. Mục tiêu là queue chạy nhiều giờ, không giả lập GPU bận hay bỏ guard để đủ 6h; thời gian thực phụ thuộc cổng và throughput. Dừng khi vượt `3.400 MiB` reserved hoặc đáy đĩa dự kiến xuống `15 GiB`; cache theo WAV ghi nguyên tử.
