# config.py

FILE_PATH = "original_data.xlsx"
FEATURES = [f"X{i}" for i in range(1, 12)] + ["PPV"]
EPOCHS = 10000
SAVE_FILE = "CRTDiff_generated_data_.xlsx"
BATCH_SIZE = 32
GEN_MULTIPLIER = 5
COND_DIST_PLOT = "condition_distribution_plot.png"
COND_DIST_FILE = "condition_distribution.xlsx"
COND_COLS = ["X1", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X11"]
