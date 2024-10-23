import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_offset(label_length):
    # 基于长度为4和8的偏移量
    base_offset_4 = 0.2
    base_offset_8 = 0.5
    
    # 在这些长度之间进行线性插值
    if label_length <= 4:
        return base_offset_4
    elif label_length >= 18:
        return base_offset_8
    else:
        # 在两个值之间插值
        return base_offset_4 + (base_offset_8 - base_offset_4) * (label_length - 4) / (8 - 4)


# 原始数据
data = {
    'Model': ['MLP', 'MLP', 'MLP', 'GBM', 'GBM', 'GBM', 'GBM'],
    'FP': ['NewFp', 'DRFP', 'NewFp', 'NewFp', 'DRFP', 'NewFp', 'DRFP'],
    'Enzyme': ['ESM-2', 'ESM-2', 'ESM1b_ts', 'ESM-2', 'ESM-2', 'ESM1b_ts', 'ESM1b_ts'],
    'Test': [0.4489, 0.4356, 0.3961, 0.4251, 0.4083, 0.3998, 0.3787],
    'Valid1': [0.41858, 0.40766, 0.36949, 0.39557, 0.38651, 0.37577, 0.36255],
    'Valid2': [0.44578, 0.43367, 0.39135, 0.41735, 0.39883, 0.40130, 0.37903],
    'Valid3': [0.44723, 0.43427, 0.39559, 0.42977, 0.41093, 0.41210, 0.39189],
    'Valid4': [0.44041, 0.42698, 0.38284, 0.41035, 0.39742, 0.38805, 0.37289],
    'Valid5': [0.41724, 0.40711, 0.37395, 0.40421, 0.38961, 0.38636, 0.36880],
}

df = pd.DataFrame(data)

# 基准配置
base = {'Model': 'MLP', 'FP': 'NewFp', 'Enzyme': 'ESM-2'}
turnup = {'Model': 'GBM', 'FP': 'DRFP', 'Enzyme': 'ESM1b_ts'}

# 生成完整的X轴标签
def create_full_label(row, base):
    full_label = []
    for col in ['Model', 'FP', 'Enzyme']:
        value = row[col]
        if value == '-':
            full_label.append(base[col])
        else:
            if value == 'ESM1b_ts':
                value = 'ESM1b$_{ESP}$'  # 使用数学文本语法
            full_label.append(value)
    return '&'.join(full_label)

# 创建X轴标签
df['Full_Label'] = df.apply(lambda row: create_full_label(row, base), axis=1)

# 重塑数据框以便于绘图
df_melted = df.melt(id_vars=['Full_Label', 'Test'], value_vars=['Valid1', 'Valid2', 'Valid3', 'Valid4', 'Valid5'], var_name='Fold', value_name='Value')

# 调整图形尺寸
plt.figure(figsize=(18, 10))  # 增大图形尺寸

# 创建箱线图
sns.boxplot(x='Full_Label', y='Value', data=df_melted, palette="Set2", width=0.3)  # 将箱体宽度减小

# 添加测试集数据点
sns.stripplot(x='Full_Label', y='Test', data=df, color='red', marker='o', size=10)

# 设置Y轴范围
plt.ylim(0.3, 0.5)

# 设置X轴标签，并手动设置标签颜色
ax = plt.gca()
labels = df['Full_Label']
ax.set_xticklabels([])  # 移除现有的X轴标签

# 替换标签颜色并调整标签位置
def colored_label(ax, labels, base, turnup):
    for i, label in enumerate(labels):
        parts = label.split('&')
        x = ax.get_xticks()[i]
        offset = 0.3  # 用于调节标签位置的偏移量，增大间距

        for part in parts:
            color = 'blue'
            if part in turnup.values():
                color = 'black'

            ax.text(x - offset, -0.05, part, ha='center', color=color, fontsize=12, transform=ax.get_xaxis_transform(), rotation=0)
            
            offset -= calculate_offset(len(part))  # 每个部分标签的偏移量，适当增加间距

colored_label(ax, labels, base, turnup)

# 选择移除或重新放置X轴标签
# ax.set_xlabel('')  # 移除X轴标签
ax.xaxis.set_label_position('top')  # 将X轴标签移至顶部

plt.title('Model Performance on Test and Cross-Validation Sets')
plt.ylabel('Performance Metric')

plt.tight_layout()
plt.show()

plt.savefig('./kcat/data/img_paper/img12/'+"ablation.png")
