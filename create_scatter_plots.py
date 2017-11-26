import pandas as pd
import matplotlib.pyplot as plt

# read our dataset from csv
df = pd.read_csv('csv_dataset_follNorm_Nov22.csv', index_col=False)

# isolate the column labels of our explanatory variables
ev_col_labels = df.columns.values
ev_col_labels = ev_col_labels[2:10]

# plot all of our numerical EV's against our class variable
# can comment out if it is not necessary
for label in ev_col_labels:
    plt.figure(1)
    plt.scatter(getattr(df, label), df.change_std, color='blue')
    plt.title("change_std as a function of " + label)
    plt.xlabel(label)
    plt.ylabel("change_std")
    plt.show()