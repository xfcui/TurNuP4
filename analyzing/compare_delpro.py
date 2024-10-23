import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ['DRFP', 'SCFP', 'DRFP_mean', 'SCFP_mean','TurNuP']
normal_results = [0.3449, 0.3481, 0.4356, 0.4489,0.3787]
noisy_results = [0.2722, 0.3075, 0.4028, 0.4308,0.3464]
impact = [(normal - noisy) / normal * 100 for normal, noisy in zip(normal_results, noisy_results)]

x = np.arange(len(labels))
width = 0.2  # 调整柱子的宽度使其变瘦

fig, ax1 = plt.subplots(figsize=(10, 6))

# 创建柱状图（展示正常结果和加噪音结果）
bars1 = ax1.bar(x - width/2, normal_results, width, label='Normal', color='skyblue')
bars2 = ax1.bar(x + width/2, noisy_results, width, label='Del_Products', color='salmon')

# 设置左边Y轴范围
ax1.set_ylim(0.1, 0.6)
ax1.set_xlim(-0.5, len(labels) - 0.5)

# 设置Y轴标签和标题
ax1.set_xlabel('model')
ax1.set_ylabel('R2')
ax1.set_title('Del_Products Experiment Results with impact Values')

# 设置X轴标签
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

# 添加图例
ax1.legend(loc='upper left')

# 显示柱状图内部的数值
for bar in bars1 + bars2:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval - 0.02, round(yval, 3), ha='center', va='bottom', color='black', fontsize=9)

# 添加竖直大括号及其数值
for i in range(len(impact)):
    # 获取每组内的两个柱状图的坐标
    left_bar = bars1[i]
    right_bar = bars2[i]

    # 计算大括号的起始和结束点
    x_bracket = (left_bar.get_x() + right_bar.get_x() + right_bar.get_width()) / 2
    y_bracket_top = left_bar.get_height()  # 左柱状图的高度
    y_bracket_bottom = right_bar.get_height()  # 右柱状图的高度

    # 绘制竖直大括号（开口向左）
    ax1.plot([x_bracket+0.05, x_bracket+0.05], [y_bracket_bottom, y_bracket_top], color='black', lw=1, linestyle='--')
    
    ax1.plot([x_bracket+0.1, x_bracket], [y_bracket_top, y_bracket_top], color='black', lw=1)
    
    ax1.plot([x_bracket+0.1, x_bracket], [y_bracket_bottom, y_bracket_bottom], color='black', lw=1)

    # 在大括号的右侧显示数值
    ax1.text(x_bracket + 0.3, (y_bracket_top + y_bracket_bottom) / 2, 
             f'{impact[i]:.1f}%', ha='right', va='center', color='darkgreen', fontsize=9)

# 调整布局和保存图像
plt.tight_layout()
plt.show()

plt.savefig('./kcat/data/img_paper/img6/'+"del_all.png")
