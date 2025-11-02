# TrafficSafeAnalyzer

一个基于 Streamlit 的交通安全分析系统，支持事故数据分析、预测模型、异常检测和策略评估。

## 功能

- 加载和清洗事故与策略数据（Excel 格式）
- 使用 ARIMA、KNN、GLM、SVR 等模型预测事故趋势
- 检测异常事故点
- 评估交通策略效果并提供推荐
- 识别事故热点路口并生成风险分级与整治建议
- 支持 AI 分析生成自然语言洞察

## 安装步骤

### 前提条件

- Python 3.8+
- Git
- 可选：Docker（用于容器化部署）

### 安装

1. 克隆仓库：

```bash
git clone https://github.com/tongnian0613/TrafficSafeAnalyzer.git
cd TrafficSafeAnalyzer
```

2. 创建虚拟环境（推荐）：

```bash
conda create -n trafficsa python=3.8 -y
conda activate trafficsa
pip install -r requirements.txt
streamlit run app.py
```

3. 安装依赖：

   (1) 基本安装（必需依赖）

   ```bash
   pip install streamlit pandas numpy matplotlib plotly scikit-learn statsmodels scipy
   ```

   (2) 完整安装（包含所有可选依赖）

   ```bash
   pip install -r requirements.txt
   ```

   (3) 或者手动安装可选依赖

   ```bash
   pip install streamlit-autorefresh openpyxl xlrd cryptography
   ```

   (4) 运行应用：

      ```bash
      streamlit run app.py
      ```

## 依赖项

列于 `requirements.txt`：

```txt
streamlit>=1.20.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
plotly>=5.0.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
scipy>=1.7.0
streamlit-autorefresh>=0.1.5
python-dateutil>=2.8.2
pytz>=2021.3
openpyxl>=3.0.9
xlrd>=2.0.1
cryptography>=3.4.7
openai>=2.0.0
```

## 配置参数

- **数据文件**：上传事故数据（`accident_file`）和策略数据（`strategy_file`），格式为 Excel；事故热点分析会直接复用事故数据，无需额外上传。
- **环境变量**（可选）：
  - `LOG_LEVEL=DEBUG`：启用详细日志
  - 示例：`export LOG_LEVEL=DEBUG`（Linux/macOS）或 `set LOG_LEVEL=DEBUG`（Windows）
- **AI 分析凭据**：应用内已预填可用的示例 API Key 与 Base URL，可直接体验；如需使用自有服务，可在侧边栏替换后即时生效。

## 示例数据

`sample/` 目录提供了脱敏示例数据，便于快速体验：

- `sample/事故/*.xlsx`：按年份划分的事故记录
- `sample/交通策略/*.xlsx`：策略发布记录

使用前建议复制到临时位置再进行编辑。

## 输入输出格式

### 输入
- **事故数据 Excel**：需包含 `事故时间`、`所在街道`、`事故类型` 列
- **策略数据 Excel**：需包含 `发布时间`、`交通策略类型` 列

### 输出
- **预测结果**：CSV 文件（例如 `arima_forecast.csv`）
- **图表**：HTML 文件（例如 `overview_series.html`）
- **策略推荐**：文本文件（`recommendation.txt`）

## 调用示例

运行 Streamlit 应用：
```bash
streamlit run app.py
```

访问 http://localhost:8501，上传数据文件并交互分析。

## 常见问题排查

**问题**：`ModuleNotFoundError: No module named 'streamlit'`  
**解决**：运行 `pip install -r requirements.txt` 或检查 Python 环境

**问题**：数据加载失败  
**解决**：确保 Excel 文件格式正确，检查列名是否匹配

**问题**：预测模型页面点击后图表未显示  
**解决**：确认干预日期之前至少有 10 条历史记录，或缩短预测天数重新提交

**问题**：热点分析提示“请上传事故数据”  
**解决**：侧边栏上传事故数据后点击“应用数据与筛选”，热点模块会复用相同数据集

## 日志分析

- **日志文件**：`logs/app.log`（需在代码中配置 logging 模块）
- **查看日志**：`tail -f logs/app.log`
- **常见错误**：
  - `ValueError`：检查输入数据格式
  - `ConnectionError`：验证网络连接或文件路径

## 升级说明

- **当前版本**：v1.0.0
- **升级步骤**：
  1. 备份数据和配置文件
  2. 拉取最新代码：`git pull origin main`
  3. 更新依赖：`pip install -r requirements.txt --upgrade`
  4. 重启应用：`streamlit run app.py`

参考 `CHANGELOG.md` 查看版本变更详情。

## 许可证

MIT License - 详见 LICENSE 文件。

[![GitHub license](https://img.shields.io/github/license/tongnian0613/repo)](https://github.com/tongnian0613/TrafficSafeAnalyzer/LICENSE)
[![Build Status](https://img.shields.io/travis/username/repo)](https://travis-ci.org/tongnian0613/repo)
