import matplotlib.pyplot as plt
import numpy as np

# Metrics extracted from logs
models = ["LSTM-FinBERT", "LSTM-CNN-FinBERT", "LSTM-LLM", "LSTM-CNN-LLM"]
rmse = [55.51, 118.92, 46.38, 74.67]
mae = [44.20, 99.23, 37.89, 58.19]
dir_acc = [50.70, 47.04, 52.96, 49.58]  # Directional Accuracy (%)
dc_prec = [15.01, 16.67, 41.36, 53.33]  # DC Precision (%)
dc_recall = [0.65, 0.65, 43.79, 20.92]  # DC Recall (%)
dc_timing = [2.75, 2.89, 0.87, 1.73]    # DC Timing Error (days)

x = np.arange(len(models))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar chart for RMSE and MAE
bars1 = ax1.bar(x - width/2, rmse, width, label='RMSE')
bars2 = ax1.bar(x + width/2, mae, width, label='MAE')

ax1.set_xlabel('Model')
ax1.set_ylabel('Error (Price Units)')
ax1.set_title('Model Performance Metrics')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.legend(loc='upper left')

# Secondary axis for percentage and timing metrics
ax2 = ax1.twinx()
ax2.plot(x, dir_acc, marker='o', linestyle='-', label='Directional Accuracy (%)')
ax2.plot(x, dc_prec, marker='s', linestyle='--', label='DC Precision (%)')
ax2.plot(x, dc_recall, marker='^', linestyle='-.', label='DC Recall (%)')
ax2.plot(x, dc_timing, marker='d', linestyle=':', label='DC Timing Error (days)')

ax2.set_ylabel('Percentage / Days')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
