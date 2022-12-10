import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

x = np.array([1, 2, 3, 4])
standard_deviation_sacle = 100

Pubmed_min = np.array([[92.8,94.3,95.2,95.7],[1.4,1,0.8,0.05]])
Computer_min = np.array([[97.9,98.3,97.9,97.1],[1.2,0.4,0.3,0.1]])
Photo_min = np.array([[95.5,96.7,96.7,97.3],[1.3,2.0,1.4,0.2]])
Weibo_min = np.array([[94.0,93.7,94.9,94.5],[1.4,0.6,0.8,0.5]])

# label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
# color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
# 线型：-  --   -.  :    ,
# marker：.  ,   o   v    <    *    +    1
plt.figure(figsize=(4, 3))
plt.grid(linestyle="--")  # 设置背景网格线为虚线
ax = plt.gca()
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框


plt.scatter(x, Pubmed_min[0],alpha=1, marker='o', color="Navy", linewidth=1.5,sizes= Pubmed_min[1]*standard_deviation_sacle)
plt.scatter(x, Computer_min[0], alpha=1, marker='<', color="Teal", linewidth=1.5,sizes= Computer_min[1]*standard_deviation_sacle)
plt.scatter(x, Photo_min[0], alpha=0.8, marker='v', color="Maroon", linewidth=1.5,sizes= Photo_min[1]*standard_deviation_sacle)
plt.scatter(x, Weibo_min[0], alpha=0.8, marker='*', color="green", linewidth=1.5,sizes= Weibo_min[1]*standard_deviation_sacle)

plt.plot(x, Pubmed_min[0],color="Navy", marker='o',linewidth=1.5, label="Pubmed Min")
plt.plot(x, Computer_min[0], color="Teal", marker='<',linewidth=1.5, label="Amazon Computer Min")
plt.plot(x, Photo_min[0], color="Maroon", marker='v',linewidth=1.5, label="Amazon Photo Min")
plt.plot(x, Weibo_min[0], color="green", marker='*',linewidth=1.5, label="Weibo")

group_labels = ['1', '2', '3', '4']  # x轴刻度的标识
plt.xticks(x, group_labels, fontsize=10, fontweight='bold') 
plt.yticks(fontsize=8, fontweight='bold')
# plt.title("example", fontsize=12, fontweight='bold') 
plt.xlabel("The Number of views", fontsize=10, fontweight='bold')
plt.ylabel("Detection rate (%)", fontsize=10, fontweight='bold')
plt.xlim(0.5, 4.5)
plt.ylim(92, 100)

plt.legend(loc=0, ncol=2, borderaxespad = 0.,bbox_to_anchor=(0.92,1.15),fontsize=5)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=8, fontweight='bold')  # 设置图例字体的大小和粗细

plt.savefig('./result/num_views.eps', format='eps', bbox_inches="tight")
plt.show()
