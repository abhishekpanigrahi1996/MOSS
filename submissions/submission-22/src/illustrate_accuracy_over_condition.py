import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data into a pandas DataFrame
data = pd.read_csv("out_vgg_imagenet.csv", header=None)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Left y-axis plot
sns.lineplot(x=0, y=1, data=data, ax=ax1, label="Y1", color="blue")
ax1.set_ylabel("Accurcay")
ax1.set_xlabel("rank reduction")
ax1.tick_params(axis="y", labelcolor="blue")

# Right y-axis plot
ax2 = ax1.twinx()
sns.lineplot(x=0, y=17, data=data, ax=ax2, label="Y2", color="red")
ax2.set_ylabel("Condition")
ax2.tick_params(axis="y", labelcolor="red")
#ax2.set_yscale("log")

# Title and legend
fig.suptitle("Accuracy vs Condition of VGG16 classifier layer 3 over rank reduction", fontsize=16)
fig.tight_layout()
plt.savefig("out_vgg_imagenet.png")
