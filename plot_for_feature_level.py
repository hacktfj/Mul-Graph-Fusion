import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.figure(figsize=(4.5,4.5))
width = 0.2
ax = plt.subplot(2,1,1)
# bwgnn, gin
label = ["pubmed","amazon_computer","amazon_photo"]
index = np.arange(len(label))
values1 = [94.6, 92.6, 90.5]
values2 = [95.9, 98.3, 95.1]
plt.bar(index, values1, width,label="No fusion",color="Green")
plt.bar(index + width, values2, width,label="Feature fusion",color="Navy")
# plt.set_xticks()
ax.set_xticks(index+0.15)
ax.set_xticklabels(label)
plt.yticks(fontweight='bold',fontsize=8)
plt.ylim([80,100])
plt.legend(loc="best", bbox_to_anchor=(0.9,1.2),borderaxespad = 0.,ncol=2,)
plt.ylabel("Detection rate (%)",fontweight='bold')

for a,b in zip(index,values1):  
 plt.text(a,b,'%.1f'%b,ha='center',va='bottom',fontsize=8)

for a,b in zip(index+width,values2):
 plt.text(a,b,'%.1f'%b,ha='center',va='bottom',fontsize=8)

ax = plt.subplot(2,1,2)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
# gat, gin, bwgnn
label = ["Pubmed","Amazon computer","Amazon photo"]
index = np.arange(len(label))
values1 = [95.5, 95.8, 90.6]
values2 = [95.8, 97.5, 96.2]
plt.bar(index, values1, width,label="No fusion",color="Green")
plt.bar(index + width, values2, width,label="Feature fusion",color="Navy")
# plt.set_xticks()
ax.set_xticks(index+0.15)
ax.set_xticklabels(label)
plt.ylim([88,100])
plt.yticks(fontweight='bold',fontsize=8)
plt.ylabel("Detection rate (%)",fontweight='bold')
plt.xlabel("Datasets", fontweight='bold',fontsize=10)
for a,b in zip(index,values1):  
 plt.text(a,b,'%.1f'%b,ha='center',va='bottom',fontsize=8)

for a,b in zip(index+width,values2):
 plt.text(a,b,'%.1f'%b,ha='center',va='bottom',fontsize=8)
plt.savefig(str('./result/feature_level.eps'), bbox_inches='tight', format='eps')
plt.show()