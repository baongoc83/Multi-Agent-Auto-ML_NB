# ML Model Lifecycle 

> Khung tham chiếu cho ML model development trong banking 

---

## 1. Overview

ML lifecycle thường được vẽ như một cycle (` business framing --> data → model → deploy`). Mỗi lần monitoring phát hiện drift hoặc performance suy giảm, quyết định đúng nên là "kiểm tra lại assumption ban đầu có còn đúng không" sau đó "retrain" và lặp lại cycle khác.

Đây là cycle 7 giai đoạn:

```mermaid
flowchart LR
    BF["Business framing<br/><i>Vấn đề, KPI, ROI</i>"]:::business
    D["Data<br/><i>Thu thập, khám phá</i>"]:::dev
    F["Features<br/><i>Tạo, lưu trữ</i>"]:::dev
    M["Modeling<br/><i>Train, tune</i>"]:::dev
    E["Evaluation<br/><i>Metrics + ROI</i>"]:::dev
    DP["Deployment<br/><i>Serve, A/B</i>"]:::prod
    MN["Monitoring<br/><i>Drift, retrain</i>"]:::prod

    BF --> D --> F --> M --> E --> DP --> MN
    MN -.->|Iterate / re-frame| BF

    classDef business fill:#FAEEDA,stroke:#854F0B,color:#412402
    classDef dev fill:#EEEDFE,stroke:#534AB7,color:#26215C
    classDef prod fill:#E1F5EE,stroke:#0F6E56,color:#04342C
```

Ba nhóm màu phản ánh ba bản chất công việc khác nhau:

| Màu | Stage | Bản chất |
|---|---|---|
| 🟡 Amber | Business framing | Ngôn ngữ stakeholder, kinh doanh, không có code |
| 🟣 Purple | Data, Features, Modeling, Evaluation | Iteration nội bộ trong dev environment |
| 🟢 Teal | Deployment, Monitoring | Model gặp dữ liệu thật và hành vi thật, thường trên live |

---

