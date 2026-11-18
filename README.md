# 设备异常推理 API

按设备型号提交振动、电流测值，按需装载固定权重的逻辑回归模型并返回异常分数，用于安排巡检。
三套模型、可配置推理延迟与字节预算通过 Docker Compose 提供，Swagger UI 可直接调用。

## 快速开始

```bash
docker compose up --build -d
curl -s localhost:8000/v1/models
open http://localhost:8000/docs          # Swagger UI
```

`POST /v1/infer` 示例：

```bash
curl -s localhost:8000/v1/infer \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"pump","vibration":[2.1,-1.9,3.4,-3.0],"current":[12.4,12.8,11.9,12.6]}'
```

```json
{
  "model_id": "pump",
  "model_name": "离心泵",
  "model_version": "1.0.0",
  "anomaly_score": 0.0312,
  "is_anomaly": false,
  "decision_threshold": 0.5,
  "features": {"vibration_mean": 0.15, "...": 0.0}
}
```

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/healthz` | 存活探针 |
| GET | `/v1/models` | 可用设备型号及权重大小 |
| POST | `/v1/infer` | 提交测值，返回异常分数 |
| GET | `/v1/cache` | 预算/驻留/预留/在途/计数快照 |
| GET | `/docs` | Swagger UI |

错误统一为 `{"error": {"code": ..., "message": ..., ...}}`：

| 状态码 | code | 场景 |
| --- | --- | --- |
| 404 | `unknown_model` | 型号不在清单中 |
| 422 | `invalid_measurement` | 空数组、非有限值、维度错误 |
| 500 | `model_load_failed` / `inference_failed` | 权重读取或校验失败 |
| 503 | `model_busy` | 预算不足且所有驻留模型都有在途推理；响应带 `Retry-After` |
| 507 | `model_too_large` | 单模型权重超过整个预算，永远不会成功 |

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_CACHE_BYTES` | `640` | 驻留权重 + 装载预留的上限（字节）。每套模型权重 296 字节，640 可容纳两套，第三套会触发 LRU 淘汰 |
| `INFERENCE_DELAY_MS` | `250` | 模拟推理耗时，便于在 Swagger 中观察并发行为 |
| `MODEL_LOAD_DELAY_MS` | `0` | 模拟权重读取耗时，便于复现并发冷装载 |
| `MODEL_DIR` | `./models` | 模型清单与权重文件目录 |
| `INFERENCE_PORT` | `8000` | Compose 暴露的主机端口 |

预算只统计数组形式的权重（`coefficients`、`feature_mean`、`feature_scale`、`intercept`）。
清单里的 `weights_bytes` 是预留依据；装载后实际字节必须与其一致，否则视为装载失败。

## 并发与内存语义

实现集中在 `app/cache.py`：

1. **总占用不超预算**：`resident_bytes + reserved_bytes <= MODEL_CACHE_BYTES` 在每次状态变更时成立。
2. **单飞装载**：冷请求在缓存锁内登记一条 `LOADING` 记录并创建独立装载任务；后续同模型请求只等待同一个事件，
   实测 10 个并发冷请求只产生 1 次 `load_attempt`。
3. **等待装载时断连**：请求方只取消自己的等待，不触碰装载任务；其他等待者仍能拿到模型。
   装载任务归缓存所有，最后一个等待者离开也不会取消它。
4. **推理中断连**：请求可以立即结束，但计算在线程池中继续，模型租约由计算任务在真正结束后释放，
   期间该模型不可被淘汰。
5. **淘汰策略**：只考虑 `inflight == 0` 且没有等待者的驻留模型，按 LRU 顺序；若这些模型的全部空间仍不够，
   在淘汰任何模型之前就返回 503，避免"淘汰了却仍然装不下"。
6. **装载失败**：删除记录并归还预留，其他模型不受影响；下次请求会重试该模型。

## 模型

`models/` 下是三份固定权重逻辑回归（`pump` / `fan` / `compressor`）与 `manifest.json`，
由 `scripts/generate_models.py` 确定性生成。特征为振动与电流各自的
均值、RMS、峰值、标准差、波峰因数、峭度，共 12 维；模型各自标准化后做点积并取 sigmoid。
服务只读取权重，不做任何在线训练。

## 本地开发与测试

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt pytest pytest-asyncio httpx pyyaml
.venv/bin/python -m pytest -q
.venv/bin/uvicorn app.api:app --reload
```

测试覆盖：特征抽取边界、清单/权重校验、预算与预留、LRU 淘汰、忙拒绝、单飞装载、
装载失败恢复，以及在 ASGI 层注入 `http.disconnect` 的两种断连语义。
