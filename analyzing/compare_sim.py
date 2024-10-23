import matplotlib.pyplot as plt
import numpy as np

# 数据：模型在不同相似度区间下的结果
similarity_intervals = ['0%-18%', '18%-36%', '36%-54%', 'Same (46%)']

# my_model_results = [0.0706,0.2498,0.248,0.5569]  # 你的模型结果示例数据
# previous_model_results = [0.0408,0.2331,0.2436,0.5638]  # 前人模型结果示例数据
# new_model_results = [0.0457, 0.2211, 0.2425, 0.5704]  # 新的橙色折线的数据

# my_model_results = [0.2056,0.3869,0.3553,0.6114]  # 你的模型结果示例数据
# previous_model_results = [0.1925,0.3609,0.3561,0.6061]  # 前人模型结果示例数据

my_model_results = [0.47706,0.62884,0.60353,0.78639]  # 你的模型结果示例数据
previous_model_results = [0.44341,0.60861,0.60321,0.78272]  # 前人模型结果示例数据



# 自定义的 y 轴比例函数
def custom_y_scale(y):
    if y < 0.3:
        return y  # 0.1到0.3区域保持线性比例
    else:
        return 0.3 + (y - 0.3) / 2  # 0.3到0.6区域压缩

# 应用自定义比例的逆函数
def inverse_custom_y_scale(y):
    if y < 0.3:
        return y
    else:
        return 0.3 + (y - 0.3) * 2

# 创建折线图
plt.figure(figsize=(8, 6))

# 你的模型（红色）
# plt.plot(similarity_intervals, [custom_y_scale(y) for y in my_model_results], marker='o', linestyle='-', color='red', label='New fp')
plt.plot(similarity_intervals, [y for y in my_model_results], marker='o', linestyle='-', color='red', label='New fp')

# 前人模型（蓝色）
# plt.plot(similarity_intervals, [custom_y_scale(y) for y in previous_model_results], marker='o', linestyle='-', color='blue', label='DRFP')
plt.plot(similarity_intervals, [y for y in previous_model_results], marker='o', linestyle='-', color='blue', label='DRFP')

# # 新的橙色折线
# plt.plot(similarity_intervals, [custom_y_scale(y) for y in new_model_results], marker='o', linestyle='-', color='green', label='New fp(GBM)')


# 设置图例
plt.legend()

# 设置标题和轴标签
plt.title('FPS Performance Across Different Similarity Intervals')
# plt.title('Mean Performance Across Different Similarity Intervals')
plt.xlabel('Reaction Similarity')
plt.ylabel('Pearson r')

# 自定义 y 轴刻度和标签
yticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# plt.yticks([custom_y_scale(y) for y in yticks], [f'{y:.1f}' for y in yticks])
plt.yticks([y for y in yticks], [f'{y:.1f}' for y in yticks])



# 显示数值
# for i, value in enumerate(my_model_results):
#     plt.text(similarity_intervals[i], custom_y_scale(value), f'{value:.2f}', ha='right', va='bottom', color='red')  # 显示在左边
for i, value in enumerate(my_model_results):
    plt.text(similarity_intervals[i], value, f'{value:.2f}', ha='right', va='bottom', color='red')  # 显示在左边



# for i, value in enumerate(previous_model_results):
#     plt.text(similarity_intervals[i], custom_y_scale(value), f'{value:.2f}', ha='left', va='bottom', color='blue')  # 显示在右边
for i, value in enumerate(previous_model_results):
    plt.text(similarity_intervals[i], value, f'{value:.2f}', ha='left', va='bottom', color='blue')  # 显示在右边



# for i, value in enumerate(new_model_results):
#     plt.text(i + 0.1, custom_y_scale(value), f'{value:.2f}', ha='left', va='top', color='green')  # 显示在右边并向右偏移


plt.ylim(0.3, 1)

# 移除网格线
plt.grid(False)

# 显示图表
plt.tight_layout()
plt.show()

plt.savefig('./kcat/data/img_paper/img8/'+"sim_fp_pearson.png")
# plt.savefig('./kcat/data/img_paper/img8/'+"sim_mean.png")











