import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('final_chosen_posteriors.csv')

sns.pairplot(df)
plt.show()