# Báo cáo Chiến dịch Nâng cấp Biểu diễn Âm học & Backbone (`BACKBONE_REP_CAMPAIGN_20261122_v1`)

- **Split mode:** `clean_11` (`test_sealed_opened=False`)
- **Git HEAD:** `aef0d8aea29b96dfec513405e7a2d533877ae71c`
- **Stage 1 Top-2 Backbone (chọn thuần trên Cal):** `wavlm_base_plus`, `None`
- **Stage 5 (thuần Cal):** `không nhánh nào đạt`
- **Trạng thái:** `COMPLETE_VERIFIED_DIAGNOSTIC`
- Eval là bộ chẩn đoán đã dùng; phần Spontaneous chính chỉ dùng lát Held-6. Không phải xác nhận tập mới.

## 1. Stage 1 — Cal-Only Ridge Kill-Gate (phá điểm mù `can_lp > -0,2`)

| Biểu diễn | Cal L2 tAP | Cal L2 Blind-Spot AP (`can_lp > -0.2`) | Cal L2 gAP | Cal L1 FA @ R70 | Composite |
|---|---:|---:|---:|---:|---:|
| `koel_only` | 0.7446 | 0.1066 | 0.3881 | 1.90% | 0.4819 |
| `koel+wavlm_large` | 0.7462 | 0.1138 | 0.3905 | 1.35% | 0.4853 |
| `koel+wavlm_base_plus` | 0.7444 | 0.1050 | 0.4132 | 1.48% | 0.4863 |
| `koel+huper_mid` | 0.7445 | 0.1139 | 0.3475 | 1.43% | 0.4759 |
| `koel+whisper_small` | 0.7441 | 0.1078 | 0.3965 | 2.20% | 0.4837 |
| `koel+wavlm_base_plus+charsiu_proto` | 0.7379 | 0.1111 | 0.4132 | 1.30% | 0.4849 |

## 2. Xác nhận 3 Seed Ghép Cặp trên Eval (Native Cal-locked, TB ± SD)

| Hệ thống | Held-6 F1 | EpaDB F1 | Spont (H6) F1 | SAA-27 F1 (tAP) | TIMIT FA/câu | Ctrl Recall |
|---|---:|---:|---:|---:|---:|---:|
| `S0_clean` | 65.32±0.70% | 68.57±0.05% | 64.68±1.25% | 43.05±1.31% (0.4664±0.0014) | 0.578±0.070 | 95.82±0.21% |

## 3. Tất cả các nhánh 1-seed sàng lọc (Stage 2–4, seed đầu tiên)

| Nhánh | Held-6 F1 | Held-6 Blind-Spot AP | EpaDB F1 | SAA-27 F1 | TIMIT FA/câu | Ctrl Recall |
|---|---:|---:|---:|---:|---:|---:|
| `A1_top1` | 64.80% | 0.1717 | 68.75% | 43.22% | 0.521 | 94.62% |
| `A3_top1_charsiu_proto` | 65.17% | 0.1718 | 69.26% | 42.30% | 0.510 | 96.77% |
| `B1_xattn_top1` | 64.46% | 0.1657 | 68.23% | 43.11% | 0.911 | 97.13% |
| `B2_xattn_dual_frame` | 64.02% | 0.1369 | 69.23% | 42.80% | 0.604 | 94.27% |
| `C1_wavlm_lora_direct` | 64.63% | 0.1701 | 69.14% | 43.48% | 1.161 | 94.98% |
| `C2_wavlm_lora_warm_af` | 64.19% | 0.1382 | 68.75% | 45.21% | 1.651 | 97.49% |

## 4. Độ trễ GPU GTX 1650 (chỉ encoder, batch 1; không phải độ trễ end-to-end)

| Backbone | Short P50 (ms) | Short P95 (ms) | Stella P50 (ms) | Peak VRAM (MiB) |
|---|---:|---:|---:|---:|
| `wavlm_base_plus` | 67.7 | 87.1 | 460.3 | 1076.0 |
| `wavlm_large` | 147.2 | 206.4 | 1175.5 | 2076.0 |

## 5. Bảo toàn & Xác minh
- Mọi nhánh MDD dùng đúng cache CTC đã khóa của Koel; không đo lại PER đầu-cuối trong lượt này.
- Toàn bộ dự đoán Eval được niêm phong SHA-256 trước khi ghép gold và kiểm đếm độc lập trên CPU.
- `test_sealed.jsonl`, `GOAL.md`, và mọi checkpoint cũ (`aef0d8a`, `a9bb69ed…`) giữ nguyên bất biến.
