import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'

#categories = ['Non-Existence', 'Unchanged', 'Substrates\n-Changed', 'Products\n-Changed']
categories = ["Substrates\n-Changed" , 'Unchanged', 'Products\n-Changed','Non-Existence']
# F1-Score数据

f1_pretrained = [0.976, 0.994, 0.977, 0.999]
f1_non_pretrained = [0.851, 0.942, 0.796, 0.994]

# 设置柱状图参数
width = 0.2  # 柱子的宽度
gap = 0.1   # 组内两个柱子之间的间隙
x = np.arange(len(categories))  # 大组的位置

fig, ax = plt.subplots(figsize=(5, 4),dpi=300)  # 调整图形的宽度和高度

# 绘制每个大组的柱状图，组内两个柱子之间留出空隙
bars1 = ax.bar(x - (width + gap)/2, f1_pretrained, width, label='Pretrained', color='blue')
bars2 = ax.bar(x + (width + gap)/2, f1_non_pretrained, width, label='Non-Pretrained', color='orange')

# 在每个柱子上方添加数值标签
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2 -0.03, bar.get_height() + 0.005, f'{bar.get_height():.3f}', 
            ha='center', va='bottom', fontsize=10)

for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2 +0.03, bar.get_height() + 0.005, f'{bar.get_height():.3f}', 
            ha='center', va='bottom', fontsize=10)

# 设置x轴标签，并调整字体大小
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9)

# 添加图例并缩小图例字体
ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1, 1.3), frameon=False)

# 设置y轴范围
ax.set_ylim(0.5, 1)

# 只保留xy轴，移除右边和上边的边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加轴标签和标题
ax.set_ylabel('F1-Score', fontsize=10)

# 自动调整布局以防止标签重叠
plt.tight_layout()
plt.subplots_adjust(top=0.75)
# 显示图表
plt.show()

plt.savefig('./kcat/data/img_paper/img2/'+"all.png",dpi=300)






















