import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def perform_binning_and_plot(df,file_name):
    df2 = df.copy()
    bins = np.arange(0, df2["los"].max(), 1)
    col = f"los_mv_bin"
    df2[col] = pd.cut(df2["los"], bins,include_lowest=True,right=False)
    df2[col] = df2[col].apply(lambda x: x.left)
    df_binned = df2[col].value_counts().sort_index()



    if not os.path.exists("./output_los"):
        os.makedirs("./output_los")
    if not os.path.exists("./figures"):
        os.makedirs("./figures") 
    df_binned.to_csv(f"./output_los/{file_name}.csv")

    df_binned.plot()
    plt.title(f"{file_name} n={len(df2)}")
    plt.xlabel("Days")
    plt.ylabel("Count")
    plt.savefig(f"./figures/{file_name}.png")
    plt.close()
    return df_binned
