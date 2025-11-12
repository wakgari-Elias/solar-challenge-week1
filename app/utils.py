import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV into a DataFrame."""
    return pd.read_csv(csv_path)

def plot_ghi_boxplot(df: pd.DataFrame, country: str):
    """Return a Seaborn boxplot for GHI for a selected country."""
    plt.figure(figsize=(8,5))
    sns.boxplot(x="Country", y="GHI", data=df[df["Country"] == country])
    plt.title(f"GHI Distribution for {country}")
    plt.tight_layout()
    return plt

def top_regions_table(df: pd.DataFrame, top_n: int = 5):
    """Return top N regions by average GHI."""
    return df.groupby("Region")["GHI"].mean().sort_values(ascending=False).head(top_n).reset_index()
