import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define the exact values from the V9 SE-Attention Hybrid inference
# Format: [[True Negatives, False Positives],
#          [False Negatives, True Positives]]
cm = np.array([
    [2272, 864],
    [758, 2378]
])

# 2. Define the class labels
class_names = ["Healthy (0.0)", "Cancer (1.0)"]

# 3. Set up the matplotlib figure
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

# 4. Create the heatmap using seaborn
# 'fmt="d"' ensures the numbers are formatted as standard integers (no scientific notation)
# 'cmap="Blues"' provides a clean, academic, publication-ready color gradient
ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                 xticklabels=class_names, yticklabels=class_names,
                 annot_kws={"size": 16, "weight": "bold"}, 
                 cbar_kws={'label': 'Number of Patients'})

# 5. Customize the labels and title for maximum readability
plt.xlabel('AI Prediction', fontsize=14, fontweight='bold', labelpad=10)
plt.ylabel('Actual Diagnosis', fontsize=14, fontweight='bold', labelpad=10)
plt.title('V9 SE-Attention Hybrid Model\nConfusion Matrix', fontsize=16, fontweight='bold', pad=20)

# Adjust tick formatting
plt.xticks(fontsize=12)
plt.yticks(fontsize=12, rotation=0)

# 6. Save the figure as a high-resolution PNG
plt.tight_layout()
file_name = 'v9_se_attention_cm.png'
plt.savefig(file_name, dpi=300, bbox_inches='tight')

print(f"✅ Success! High-resolution confusion matrix saved as: {file_name}")
