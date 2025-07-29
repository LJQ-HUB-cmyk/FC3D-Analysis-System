# D数据处理与分析系统 v1.0

## 项目简介

这是一个基于机器学习和统计分析的福彩3D彩票数据分析与预测系统。系统整合了数据获取、特征工程、机器学习建模、统计分析和微信推送等功能，为3D提供数据驱动的号码推荐。

## 主要功能

### 1. fc3d_data_processor.py - 数据获取与预处理
- **数据获取**：从网络文本文件获取福彩3D历史开奖数据
- **数据清洗**：智能解析和验证3D号码格式
- **数据整合**：将新获取的数据与本地CSV文件合并
- **错误处理**：多种编码支持，网络连接异常处理

### 2. fc3d_analyzer.py - 数据分析与预测核心引擎
- **特征工程**：
  - 基本统计特征（和值、跨度、奇偶分布）
  - 质数特征、大小数特征
  - 形态特征（豹子、组三、组六）
  - 重号特征（与上期重复数字个数）

- **统计分析**：
  - 频率和遗漏分析
  - 模式识别（最常见的和值、跨度等）
  - 关联规则挖掘（Apriori算法）

- **机器学习**：
  - LightGBM二分类模型（预测每个位置每个数字的出现概率）
  - 滞后特征创建
  - 交互特征生成
  - 并行模型训练

- **评分系统**：
  - 综合频率、遗漏、ML预测概率的加权评分
  - 模式匹配奖励机制
  - 归一化处理

- **策略回测**：
  - 历史数据验证
  - 投资回报率计算
  - 中奖等级统计

- **参数优化**：
  - Optuna框架自动参数调优
  - 多目标优化
  - 实时进度监控

### 3. fc3d_bonus_calculation.py - 奖金计算与验证
- **自动验证**：读取最新开奖数据验证推荐效果
- **中奖计算**：支持直选、组三、组六三种玩法
- **报告生成**：详细的中奖分析报告
- **历史记录**：自动管理评估记录

### 4. fc3d_wxpusher.py - 微信推送功能
- **推荐推送**：发送预测结果到微信
- **验证推送**：发送中奖验证报告
- **系统监控**：错误通知和日报功能
- **多格式支持**：摘要版和完整版推送

## 系统特色

### 🤖 智能化分析
- 基于LightGBM的机器学习预测
- 多维度特征工程
- 自适应参数优化

### 📊 全面统计
- 历史频率和遗漏分析
- 模式识别和关联规则挖掘
- 回测验证和性能评估

### 🎯 精准推荐
- 多因子加权评分系统
- 模式匹配奖励机制
- 组合多样性控制

### 📱 实时推送
- 微信实时推送预测结果
- 自动验证和回报分析
- 系统状态监控

## 安装配置

### 环境要求
- Python 3.8+
- 依赖包见 requirements.txt

### 安装步骤
```bash
# 克隆项目
git clone <repository-url>
cd 3D-LightGBM-Log-SVC-v1.0

# 安装依赖
pip install -r requirements.txt

# 配置微信推送（可选）
# 编辑 fc3d_wxpusher.py 中的配置信息
```

### 微信推送配置
在 `fc3d_wxpusher.py` 中配置：
```python
APP_TOKEN = "your_wxpusher_app_token"
USER_UIDS = ["your_user_uid"]
TOPIC_IDS = [your_topic_id]
```

## 使用指南

### 基本使用流程

1. **数据获取**
```bash
python fc3d_data_processor.py
```

2. **分析预测**
```bash
python fc3d_analyzer.py
```

3. **验证计算**
```bash
python fc3d_bonus_calculation.py
```

### 运行模式

#### 分析模式（默认）
- 使用内置权重快速分析
- 生成推荐号码和详细报告
- 适合日常使用

#### 优化模式
在 `fc3d_analyzer.py` 中设置：
```python
ENABLE_OPTUNA_OPTIMIZATION = True
```
- 自动参数优化
- 耗时较长但效果更佳
- 适合策略调优

### 输出文件

- `fc3d.csv` - 原始开奖数据
- `fc3d_processed.csv` - 预处理后的特征数据
- `fc3d_analysis_output_YYYYMMDD_HHMMSS.txt` - 详细分析报告
- `latest_fc3d_analysis.txt` - 最新分析报告
- `latest_fc3d_calculation.txt` - 验证计算报告

## 算法原理

### 数据预处理
1. **号码解析**：将3位数字分解为百位、十位、个位
2. **特征构建**：计算和值、跨度、奇偶分布等衍生特征
3. **滞后特征**：使用历史期数据作为预测特征

### 统计分析
1. **频率分析**：统计每个数字在各位置的出现频率
2. **遗漏分析**：计算当前遗漏、平均遗漏、最大历史遗漏
3. **模式识别**：识别最常见的和值、跨度、形态等模式

### 机器学习建模
1. **模型选择**：LightGBM二分类模型
2. **特征工程**：滞后特征 + 交互特征
3. **并行训练**：每个位置每个数字独立训练模型
4. **概率预测**：输出每个数字的出现概率

### 评分融合
综合评分 = 频率分 + 遗漏分 + 最大遗漏比率分 + 近期频率分 + ML预测分

### 组合生成
1. **候选筛选**：每个位置选取评分最高的N个数字
2. **组合生成**：笛卡尔积生成所有可能组合
3. **模式奖励**：匹配历史最常见模式给予奖励
4. **多样性控制**：确保推荐组合的多样性

## 性能指标

### 回测指标
- **投资回报率(ROI)**：基于历史回测的收益率
- **中奖分布**：各等级奖项的命中统计
- **命中率**：预测准确度指标

### 优化指标
- **Optuna优化得分**：综合考虑各等级奖项权重的评分
- **参数收敛性**：优化过程的稳定性

## 注意事项

1. **数据依赖**：需要稳定的网络连接获取开奖数据
2. **计算资源**：机器学习训练需要一定的计算时间
3. **参数调优**：建议定期运行优化模式更新参数
4. **理性投注**：系统仅供参考，投注需理性

## 技术栈

- **数据处理**：pandas, numpy
- **机器学习**：LightGBM, scikit-learn
- **统计分析**：mlxtend (关联规则)
- **参数优化**：Optuna
- **网络请求**：requests, BeautifulSoup
- **微信推送**：WxPusher API

## 更新日志

### v1.0 (2025-01-XX)
- 初始版本发布
- 完整的数据分析和预测功能
- 微信推送支持
- 自动验证和回测功能

## 免责声明

本系统仅用于技术研究和学习交流，不构成任何投注建议。彩票投注存在风险，请理性对待，量力而行。开发者不对使用本系统产生的任何损失承担责任。

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目：
- Bug 报告
- 功能建议  
- 代码优化
- 文档完善

## GitHub 部署指南

### 同步项目到 GitHub 仓库

1. **创建 GitHub 仓库**
```bash
# 在GitHub网站上创建新仓库，例如：FC3D-Analysis-System
```

2. **初始化本地Git仓库**
```bash
# 进入项目目录
cd 3D-LightGBM-Log-SVC-v1.0

# 初始化Git仓库
git init

# 添加所有文件到暂存区
git add .

# 提交初始版本
git commit -m "🎯 初始提交: 福彩3D分析系统 v1.0"

# 设置默认分支为main
git branch -M main
```

3. **连接远程仓库**
```bash
# 添加远程仓库（替换为你的仓库URL）
git remote add origin https://github.com/yourusername/FC3D-Analysis-System.git

# 推送到远程仓库
git push -u origin main
```

4. **配置GitHub Actions（自动化运行）**
```bash
# GitHub Actions工作流文件已包含在项目中
# 文件位置: .github/workflows/daily-analysis.yml
# 将在每天北京时间上午8点自动运行分析
```

### 配置微信推送（可选）

1. **获取WxPusher配置**
   - 访问 [WxPusher官网](https://wxpusher.zjiecode.com/)
   - 获取 APP_TOKEN
   - 获取用户 UID 或创建主题 ID

2. **在GitHub中设置Secrets（推荐）**
```bash
# 在GitHub仓库页面 Settings → Secrets and variables → Actions 中添加：
WXPUSHER_APP_TOKEN=your_app_token
WXPUSHER_USER_UIDS=your_user_uid
WXPUSHER_TOPIC_IDS=your_topic_id
```

3. **或直接修改代码配置**
```python
# 编辑 fc3d_wxpusher.py 文件
APP_TOKEN = "your_wxpusher_app_token"
USER_UIDS = ["your_user_uid"]
TOPIC_IDS = [your_topic_id]
```

### 本地开发流程

1. **日常更新代码**
```bash
# 拉取最新代码
git pull origin main

# 修改代码后提交
git add .
git commit -m "📊 更新: 描述你的修改"
git push origin main
```

2. **手动运行分析**
```bash
# 按顺序运行三个脚本
python fc3d_data_processor.py    # 数据获取
python fc3d_analyzer.py          # 分析预测
python fc3d_bonus_calculation.py # 验证计算
```

3. **查看运行日志**
```bash
# 查看最新分析报告
cat latest_fc3d_analysis.txt

# 查看验证计算结果
cat latest_fc3d_calculation.txt
```

### GitHub Actions 监控

- **运行时间**: 每天北京时间上午8:00自动执行
- **运行内容**: 数据获取 → 分析预测 → 验证计算 → 微信推送
- **查看日志**: 在GitHub仓库的 Actions 标签页查看运行状态
- **文件更新**: 系统会自动更新数据文件并提交到仓库

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。
