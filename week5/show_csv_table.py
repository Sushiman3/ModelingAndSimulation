# %%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('price_calorie.csv')
print(df)

# 表を画像として保存
fig, ax = plt.subplots(figsize=(8, len(df)*0.5+1))
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(df.columns))))
plt.tight_layout()
plt.savefig('sample_data2_table.png', dpi=200)
plt.show()
# %%