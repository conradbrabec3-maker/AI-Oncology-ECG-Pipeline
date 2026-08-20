import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Your exact V4 numbers
# Layout: [[True Negatives, False Positives],
#          [False Negatives, True Positives]]
cm_data = np.array([
    [6938, 1298],
    [161, 8083]
])

# 2. Define the medical labels
labels = ['Healthy (0)', 'Cancer (1)']

# 3. Set up the visual style of the plot
plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    cm_data, 
    annot=True,        # Show the numbers inside the boxes
    fmt='d',           # Format as standard integers (no scientific notation)
    cmap='Blues',      # Use a professional blue color scale
    xticklabels=labels, 
    yticklabels=labels,
    annot_kws={"size": 16} # Make the numbers large and readable
)

# 4. Add titles and axis labels
plt.title('AI Oncology V4 Confusion Matrix', fontsize=18, pad=20, fontweight='bold')
plt.xlabel('AI Prediction', fontsize=14, fontweight='bold', labelpad=15)
plt.ylabel('Actual Diagnosis', fontsize=14, fontweight='bold', labelpad=15)

# 5. Save the image as a high-res PNG file
output_filename = 'v4_confusion_matrix.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"✅ Success! The confusion matrix image has been saved as '{output_filename}'")
