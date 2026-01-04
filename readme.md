# TrafficSafeAnalyzer

基于 Streamlit 的交通安全分析系统，支持事故数据分析、多模型预测、异常检测、策略评估和 AI 智能分析。

## 功能特性

### 核心功能模块

| 模块 | 功能说明 |
|------|----------|
| 总览 | 可视化事故趋势、KPI 指标展示（今日/本周事故数、预测偏差、策略覆盖率等） |
| 事故热点 | 识别高发路口，生成风险分级与整治建议 |
| AI 分析 | 基于 DeepSeek API 生成专业分析报告和改进建议 |
| 预测模型 | 支持 ARIMA、KNN、GLM、SVR 等多模型预测对比 |
| 模型评估 | 对比各模型预测效果（RMSE、MAE 等指标） |
| 异常检测 | 基于 Isolation Forest 算法检测异常事故点 |
| 策略评估 | 评估单一交通策略实施效果 |
| 策略对比 | 多策略效果横向对比分析 |
| 情景模拟 | 模拟策略上线对事故趋势的影响 |

### 技术亮点

- 支持实时自动刷新
- 交互式 Plotly 图表
- 多格式数据导出（CSV、HTML）
- Docker 容器化部署
- 中文分词支持（jieba）

## 项目结构

```
TrafficSafeAnalyzer/
├── app.py                 # 主应用入口
├── services/              # 业务逻辑层
│   ├── forecast.py        # 预测模型（ARIMA、KNN、GLM、SVR）
│   ├── hotspot.py         # 热点分析
│   ├── io.py              # 数据加载与清洗
│   ├── metrics.py         # 模型评估指标
│   └── strategy.py        # 策略评估
├── ui_sections/           # UI 组件层
│   ├── overview.py        # 总览页面
│   ├── forecast.py        # 预测页面
│   ├── model_eval.py      # 模型评估页面
│   ├── strategy_eval.py   # 策略评估页面
│   └── hotspot.py         # 热点分析页面
├── config/
│   └── settings.py        # 配置参数
├── docs/                  # 文档
│   ├── install.md         # 安装指南
│   └── usage.md           # 使用说明
├── Dockerfile             # Docker 配置
├── requirements.txt       # Python 依赖
└── environment.yml        # Conda 环境配置
```

## 安装步骤

### 前提条件

- Python 3.8+（推荐 3.12）
- Git
- 可选：Docker（用于容器化部署）

### 方式一：本地安装

1. 克隆仓库：

```bash
git clone https://github.com/tongnian0613/TrafficSafeAnalyzer.git
cd TrafficSafeAnalyzer
```

2. 创建虚拟环境（推荐）：

```bash
# 使用 conda
conda create -n trafficsa python=3.12 -y
conda activate trafficsa

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

4. 运行应用：

```bash
streamlit run app.py
```

### 方式二：Docker 部署

```bash
# 构建镜像
docker build -t trafficsafeanalyzer .

# 运行容器
docker run --rm -p 8501:8501 trafficsafeanalyzer
```

访问 `http://localhost:8501` 即可使用。

如需挂载本地数据目录：

```bash
docker run --rm -p 8501:8501 \
  -v "$(pwd)/data:/app/data" \
  trafficsafeanalyzer
```

自定义端口：

```bash
docker run --rm -p 8080:8501 \
  -e STREAMLIT_SERVER_PORT=8501 \
  trafficsafeanalyzer
```

## 依赖项

### 核心依赖

| 包名 | 版本要求 | 用途 |
|------|----------|------|
| streamlit | >=1.20.0 | Web 应用框架 |
| pandas | >=1.3.0 | 数据处理 |
| numpy | >=1.21.0 | 数值计算 |
| matplotlib | >=3.4.0 | 静态图表 |
| plotly | >=5.0.0 | 交互式图表 |
| scikit-learn | >=1.0.0 | 机器学习模型 |
| statsmodels | >=0.13.0 | 统计模型（ARIMA） |

### 可选依赖

| 包名 | 用途 |
|------|------|
| scipy | 统计检验（t-test、Mann-Whitney U） |
| streamlit-autorefresh | 页面自动刷新 |
| openpyxl / xlrd | Excel 文件读写 |
| openai | AI 分析（兼容 DeepSeek API） |
| jieba | 中文分词 |
| cryptography | 安全加密 |

## 使用说明

### 数据格式要求

**事故数据 Excel**：

| 必需列 | 说明 |
|--------|------|
| 事故时间 | 事故发生时间 |
| 所在街道 | 事故地点 |
| 事故类型 | 事故分类 |

可选列：`region`（区域）、严重程度等

**策略数据 Excel**：

| 必需列 | 说明 |
|--------|------|
| 发布时间 | 策略发布日期 |
| 交通策略类型 | 策略分类 |

### 基本操作流程

1. 启动应用后，在左侧边栏上传事故数据和策略数据（Excel 格式）
2. 设置全局筛选器：区域、时间范围、策略类型
3. 点击"应用数据与筛选"按钮加载数据
4. 在顶部标签页切换不同功能模块进行分析

### AI 分析配置

系统使用 DeepSeek API 进行 AI 智能分析：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| API Key | 预填示例密钥 | 可在侧边栏替换为自有密钥 |
| Base URL | `https://api.deepseek.com` | DeepSeek API 地址 |

AI 分析功能可生成：
- 核心指标洞察
- 策略绩效评估
- 短期/中期/长期优化建议

### 输出文件

| 类型 | 文件名示例 | 说明 |
|------|------------|------|
| 预测结果 | `arima_forecast.csv` | ARIMA 模型预测数据 |
| 模型评估 | `model_evaluation.csv` | 各模型指标对比 |
| 异常检测 | `anomalies.csv` | 异常日期列表 |
| 策略对比 | `strategy_compare.csv` | 策略效果对比表 |
| 交互图表 | `simulation.html` | Plotly 图表导出 |

## 配置参数

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `LOG_LEVEL` | 日志级别 | INFO |
| `STREAMLIT_SERVER_PORT` | 服务端口 | 8501 |
| `STREAMLIT_SERVER_HEADLESS` | 无头模式 | true（Docker 中） |

### 模型参数

配置文件：`config/settings.py`

```python
# ARIMA 参数搜索范围
ARIMA_P = range(0, 4)
ARIMA_D = range(0, 2)
ARIMA_Q = range(0, 4)

# 预测与评估
DEFAULT_HORIZON_PREDICT = 30  # 默认预测天数
DEFAULT_HORIZON_EVAL = 14     # 默认评估窗口
MIN_PRE_DAYS = 5              # 最小历史数据天数
MAX_PRE_DAYS = 120            # 最大历史数据天数

# 异常检测
ANOMALY_N_ESTIMATORS = 50     # Isolation Forest 估计器数量
ANOMALY_CONTAMINATION = 0.10  # 预期异常比例
```

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| `ModuleNotFoundError` | 运行 `pip install -r requirements.txt` |
| 数据加载失败 | 检查 Excel 文件格式，确保包含必需列名 |
| 预测图表未显示 | 确保干预日期前至少有 10 条历史数据 |
| AI 分析无响应 | 检查 API Key 有效性及网络连接 |
| 热点分析提示无数据 | 先上传事故数据并点击"应用数据与筛选" |

## 更新日志

参见 [CHANGELOG.md](CHANGELOG.md)

**当前版本**：v1.3.0

### v1.3.0 主要更新

- 集成 DeepSeek AI 分析功能（流式输出）
- 新增事故热点分析模块
- 优化预测模型性能
- 支持 Docker 容器化部署
- 改进数据可视化交互体验
- 修复多标签页导航状态问题

## 升级指南

```bash
# 备份现有数据
cp -r data data_backup

# 拉取最新代码
git pull origin main

# 更新依赖
pip install -r requirements.txt --upgrade

# 重启应用
streamlit run app.py
```

## 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 贡献

欢迎提交 Issue 和 Pull Request。

---

[![GitHub license](https://img.shields.io/github/license/tongnian0613/TrafficSafeAnalyzer)](https://github.com/tongnian0613/TrafficSafeAnalyzer/blob/main/LICENSE)
