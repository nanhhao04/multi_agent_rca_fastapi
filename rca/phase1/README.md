# Phase 1 — Phân Tích Nguyên Nhân Gốc (RCA) theo GALA

## Tổng quan

Phase 1 thực hiện **Root Cause Analysis** cho hệ thống microservices bằng 2 nguồn dữ liệu độc lập, dựa trên framework **GALA** (Graph-based Anomaly Localization Algorithm):

| File | Nguồn dữ liệu | Thuật toán | Output |
|------|---------------|------------|--------|
| `metric_prometheus.py` | Prometheus metrics | Cross-Correlation + PageRank | `rca/log/metric_rca_ranking.json` |
| `trace_jaeger.py` | Jaeger traces | TWIST scoring | `rca/log/trace_rca_ranking.json` |

---

## 1. Metric-Based RCA (`metric_prometheus.py`)

### Pipeline

```
Prometheus → Fetch 3 metrics → Z-score Anomaly → Causal DAG → PageRank → Ranking
```

### Bước 1: Thu thập metrics

Query 3 loại metric từ Prometheus qua `rate()`:

- **Request rate**: `rate(calls_total[1m])` — tần suất request
- **Error rate**: `rate(calls_total{status_code="STATUS_CODE_ERROR"}[1m]) / rate(calls_total[1m])`
- **Latency**: `rate(duration_milliseconds_sum[1m]) / rate(duration_milliseconds_count[1m])`

Mỗi metric → 1 DataFrame (cột = service, hàng = timestep).

### Bước 2: Phát hiện bất thường (Z-score)

Với mỗi service và metric, tính **rolling z-score**:

```
z(t) = |x(t) − μ_rolling| / σ_rolling
```

- `μ_rolling`, `σ_rolling`: mean và std trên cửa sổ trượt 10 timesteps
- Service bị đánh dấu **anomalous** nếu `z > 2.0` ở **bất kỳ** metric nào
- **Severity** = `min(1, max_z / 6)` — chuẩn hóa về [0, 1]

### Bước 3: Xây dựng đồ thị nhân quả (Causal DAG)

Sử dụng **lagged cross-correlation** (thay vì PC algorithm):

```
corr(A(t), B(t + lag))   với lag = 1, 2, ..., 5
```

- Nếu correlation > 0.3 tại lag > 0 với p-value < 0.05 → thêm cạnh **A → B** (A gây ra B)
- Tính correlation trên **tất cả** metrics, lấy trung bình
- Loại bỏ chu trình (cycle) bằng cách xóa cạnh yếu nhất → đảm bảo **DAG**

### Bước 4: Xếp hạng (Personalized PageRank)

Chạy trên **đồ thị đảo ngược** (reversed DAG):

- **Personalization vector**: node anomalous → seed cao hơn
- **PageRank** (α=0.85) + **Random Walk with Restart** (α=0.95)
- **Score** = `0.6 × graph_score + 0.4 × severity`

→ Service có score cao nhất = **root cause tiềm năng**

---

## 2. Trace-Based RCA (`trace_jaeger.py`)

### Pipeline

```
Jaeger → Fetch traces → Build trace DAGs → TWIST scoring → Ranking
```

### TWIST Framework

TWIST tính điểm cho mỗi service dựa trên **4 thành phần**, tất cả chuẩn hóa về [0, 1]:

#### c1: Self-Anomaly Score (w=0.30)

```
c1 = số span bất thường của service / tổng span của service
```

- Ngưỡng động: `threshold = mean + 2 × std` (học từ dữ liệu)
- Span có `duration > threshold` → bất thường

#### c2: Trace Impact Score (w=0.25)

```
c2 = số trace lỗi chứa service / tổng trace lỗi
```

- Trace lỗi = trace có ít nhất 1 span với tag `error=true` hoặc `http.status_code ≥ 500`

#### c3: Blast Radius Score (w=0.25)

```
c3 = số downstream nodes trung bình / max downstream toàn hệ thống
```

- Đo khả năng lan truyền lỗi qua DAG
- Service có nhiều con cháu → blast radius cao

#### c4: Delay Severity Score (w=0.20)

```
c4 = max |duration - mean| của service / max deviation toàn hệ thống
```

- Đo độ nghiêm trọng của latency spike

#### Công thức tổng hợp

```
score(s) = 0.30×c1 + 0.25×c2 + 0.25×c3 + 0.20×c4
```

---

## Output

Cả 2 pipeline đều lưu kết quả vào `rca/log/`:

```
rca/log/
├── metric_prometheus.csv         # raw request rate time-series
├── metric_rca_ranking.json       # metric RCA: ranking + anomalies + graph
├── trace_rca_ranking.json        # trace RCA: TWIST ranking
└── traces.json                   # raw Jaeger traces
```

## Cách chạy

```bash
# Metric-based RCA (cần Prometheus đang chạy)
cd rca/phase1
python metric_prometheus.py

# Trace-based RCA (cần Jaeger đang chạy)
python trace_jaeger.py
```

## Dependencies

```
requests, numpy, pandas, networkx, scipy
```
