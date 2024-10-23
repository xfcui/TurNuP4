import matplotlib.pyplot as plt
import seaborn as sns

# 示例数据
# my_test_result = 0.49  # 我的测试集结果
# previous_test_result = 0.436  # 前人的测试集结果

# my_test_result = 0.7  # 我的测试集结果
# previous_test_result = 0.67  # 前人的测试集结果

my_test_result = 0.77  # 我的测试集结果
previous_test_result = 0.808  # 前人的测试集结果


# my_cross_val_results = [0.4297,0.4315,0.4981,0.4857,0.4361]  # 我的5折交叉验证结果
# previous_cross_val_results = [0.40333530789389016,0.38353482393964156,0.39189245852320176,0.42044564365386916,0.348869637]  # 前人的5折交叉验证结果

# my_cross_val_results = [0.6681,0.6556,0.6355,0.635,0.6714]  # 我的5折交叉验证结果
# previous_cross_val_results = [0.6427432566631278,0.6226108619000825,0.6371590251884234,0.6630733248247811,0.592004879]  # 前人的5折交叉验证结果

my_cross_val_results = [0.7605,0.7822,0.7428,0.8125,0.8172]  # 我的5折交叉验证结果
previous_cross_val_results = [0.7840422477361155,0.7768955691409662,0.8045088804939567,0.9293225562214953,0.984529031]  # 前人的5折交叉验证结果



# 创建绘图
plt.figure(figsize=(6, 6))  # 调整图表的宽度以使其“瘦”一些

# 调整数据顺序，先展示前人的结果，再展示你的结果
cross_val_data = [previous_cross_val_results, my_cross_val_results]

# 颜色方案
box_colors = ['lightblue', 'lightcoral']  # 颜色变浅
point_colors = ['darkblue', 'darkred']
box_edge_color = 'black'  # 边框颜色保持黑色

# 绘制箱图，并调整箱体的宽度、边框线宽和颜色
sns.boxplot(data=cross_val_data, palette=box_colors, width=0.3, medianprops={'color': 'black'},
            boxprops=dict(edgecolor=box_edge_color, linewidth=0.8), # 边框颜色和线宽
            whiskerprops=dict(color=box_edge_color, linewidth=0.8), # 胡须颜色和线宽
            capprops=dict(color=box_edge_color, linewidth=0.8)) # 顶部和底部颜色和线宽


# 绘制测试集结果的散点图，并设置颜色
plt.scatter(0, previous_test_result, color=point_colors[0], s=50, label='Previous Test Result')
plt.scatter(1, my_test_result, color=point_colors[1], s=50, label='My Test Result')

# 绘制从测试集点到纵坐标轴的虚线（横向延申）
plt.plot([0, -0.2], [previous_test_result, previous_test_result], color=point_colors[0], linestyle='--', linewidth=1)
plt.plot([1, 1.2], [my_test_result, my_test_result], color=point_colors[1], linestyle='--', linewidth=1)

# 在纵坐标轴交界处显示数值
plt.text(-0.2, previous_test_result, f'{previous_test_result:.2f}', ha='right', va='center', color=point_colors[0])
plt.text(1.2, my_test_result, f'{my_test_result:.2f}', ha='left', va='center', color=point_colors[1])

# 设置X轴标签
plt.xticks([0, 1], ['Previous Results', 'My Results'])

# 设置纵坐标范围
# plt.ylim(0, 0.7)
# plt.ylim(0.3, 1)
plt.ylim(0.3, 1.2)

# 添加图例
plt.legend()

# 设置图表标题和标签
# plt.title('Comparison of Test Set Results and 5-Fold Cross-Validation Results')
# plt.xlabel('Results')
plt.ylabel('MSE')

plt.savefig('./kcat/data/img_paper/img3/'+"compare_MSE.png")






































