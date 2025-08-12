[config.ini](../config.ini)修改配置
run_parameter_analysis.py (控制层)
    ↓ 调用
NSFNet_with_logging.py (执行层)
    ↓ 生成
参数变化历史数据
    ↓ 输入到
analyze_parameter_changes.py (分析层)


📊 性能对比分析
算法	计算复杂度	主要操作	相对速度
KSP_FF	O(K)	简单路径查找 + 首次适应	1x (基准)
Random_Fit	O(P)	随机路径选择	1.2x
KSP_LB	O(K × L)	负载计算	2x
FA_CA	O(E × S)	边代价计算	5x
KSP_FA	O(K × M × L × S)	嵌套碎片化计算	10-20x
🔍 具体性能瓶颈量化
假设典型场景：
K = 3条路径
每条路径平均4跳
每条路径平均15个可能的频谱块
每个频谱集合50个槽位
KSP_FA单次请求计算量：
对比KSP_FF：
性能差距：27,000 ÷ 12 = 2,250倍！
🚨 问题严重性
在100时隙 × 20请求/时隙 = 2000请求的仿真中：
KSP_FF: 2000 × 12 = 24,000次操作
KSP_FA: 2000 × 27,000 = 54,000,000次操作
这解释了为什么KSP_FA会卡住或运行极慢！