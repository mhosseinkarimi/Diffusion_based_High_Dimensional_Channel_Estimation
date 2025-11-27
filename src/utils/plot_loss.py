import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


loss = pd.read_csv('logs/ddpm_channel_estimation_version_5.csv')
plt.plot(loss['Step'], loss['Value'])
plt.xlabel('Step')
plt.ylabel('Loss')
plt.show()