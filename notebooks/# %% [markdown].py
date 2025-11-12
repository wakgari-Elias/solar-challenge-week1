# %% [markdown]
# # Benin-Malanville Solar Data – Full EDA & Cleaning
# **Branch:** `eda-benin` | **Notebook:** `benin_eda.ipynb` | Date: November 09, 2025

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from windrose import WindroseAxes
%matplotlib inline

# ABSOLUTE PATH — GUARANTEED TO WORK
df = pd.read_csv(r'C:\Users\Y\solar-challenge-week1\data\benin-malanville.csv', parse_dates=['Timestamp'])
df = df.set_index('Timestamp')
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
df.head()

# %% [markdown]
# # Benin-Malanville Solar Data – Full EDA & Cleaning
# **Task 2** | Branch: `eda-benin` | **525,600 rows loaded**

# %% [markdown]
# ## 1. Summary Statistics & Missing-Value Report

# %% [markdown]
# 

# %%
# Summary stats
display(df.describe())

# Missing values
missing = df.isna().sum()
print("\nMissing Values:")
display(missing.to_frame('Count'))

# >5% missing?
high_missing = missing[missing > 0.05 * len(df)]
print(f"\nColumns with >5% missing: {high_missing.index.tolist() if not high_missing.empty else 'None'}")

# %% [markdown]
# ## 2. Outlier Detection & Basic Cleaning

# %%
import os

# === OUTLIER DETECTION ===
key_cols = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust']

for col in key_cols:
    df[col + '_z'] = np.abs(stats.zscore(df[col], nan_policy='omit'))

outliers = df[[c + '_z' for c in key_cols]].gt(3).any(axis=1)
print(f"Outliers flagged: {outliers.sum()} rows")

# === IMPUTE & CLEAN ===
for col in key_cols:
    df[col] = df[col].fillna(df[col].median())

df_clean = df[~outliers].copy()
df_clean = df_clean.drop(columns=['Comments'] + [c + '_z' for c in key_cols])

# === SAVE WITH RAW STRING ===
save_path = r'C:\Users\Y\solar-challenge-week1\data\benin-malanville_clean.csv'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df_clean.to_csv(save_path, index=False)

print("CLEANED & SAVED!")
print(f"   Rows: {df_clean.shape[0]} | Cols: {df_clean.shape[1]}")
print(f"   Removed: {525600 - df_clean.shape[0]} outliers + 1 column")

# %% [markdown]
# ## 3. Time Series Analysis

# %%
# === 1. FULL TIME SERIES (GHI, DNI, DHI, Tamb) ===
# Sample first 5000 rows to avoid lag
sample = df_clean[['GHI', 'DNI', 'DHI', 'Tamb']].iloc[:5000]

sample.plot(figsize=(14, 6), alpha=0.8)
plt.title('Solar Irradiance & Temperature Over Time (First 5000 Points)')
plt.ylabel('Value (W/m² or °C)')
plt.xlabel('Timestamp')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Insight: Night = negative GHI/DNI, Tamb stable

# %%
# === 2. MONTHLY AVERAGES (GHI, DNI, DHI) ===
monthly = df_clean[['GHI', 'DNI', 'DHI']].resample('M').mean()

monthly.plot(kind='bar', figsize=(12, 5), width=0.8)
plt.title('Monthly Average Solar Irradiance')
plt.ylabel('Irradiance (W/m²)')
plt.xlabel('Month')
plt.legend(title='Variable')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.show()

# Insight: Peak in dry season (Mar–May), low in rainy season

# %%
# === 3. HOURLY PATTERN (All Solar Variables) ===
hourly = df_clean.groupby(df_clean.index.hour)[['GHI', 'DNI', 'DHI']].mean()

hourly.plot(kind='line', figsize=(12, 5), marker='o')
plt.title('Average Hourly Solar Irradiance')
plt.xlabel('Hour of Day')
plt.ylabel('Irradiance (W/m²)')
plt.legend(title='Variable')
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 24))
plt.show()

# Insight: GHI peaks ~12–14h, DNI higher in clear sky, DHI high when cloudy

# %%
# === 4. ANOMALY DETECTION: NEGATIVE GHI AT NOON ===
noon_ghi = df_clean.between_time('11:00', '13:00')['GHI']
anomalies = noon_ghi[noon_ghi < 0]

print(f"Anomalies: {len(anomalies)} negative GHI values at solar noon")
if len(anomalies) > 0:
    print("Possible sensor errors or extreme cloud cover")

# %% [markdown]
# ## 4. Cleaning Impact

# %%
df_clean.groupby('Cleaning')[['ModA', 'ModB']].mean().plot(kind='bar', figsize=(8,4))
plt.title('Sensor Readings: Before vs After Cleaning')
plt.ylabel('Average Reading')
plt.show()

# %% [markdown]
# ## 5. Correlation & Relationship Analysis

# %%
# =============================================
# CORRELATION & RELATIONSHIP ANALYSIS (ONE CELL)
# =============================================
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. CORRELATION HEATMAP ===
corr_cols = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB']
corr_matrix = df_clean[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0, 
            square=True, 
            fmt='.2f',
            cbar_kws={'label': 'Correlation'})
plt.title('Correlation Heatmap: Solar & Module Temperature')
plt.tight_layout()
plt.show()

# === 2. WIND vs GHI (3 SCATTERS) ===
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.scatterplot(x='WS', y='GHI', data=df_clean, alpha=0.6, color='blue', ax=axes[0])
axes[0].set_title('Wind Speed vs GHI'); axes[0].set_xlabel('WS (m/s)'); axes[0].set_ylabel('GHI')

sns.scatterplot(x='WSgust', y='GHI', data=df_clean, alpha=0.6, color='green', ax=axes[1])
axes[1].set_title('Gust Speed vs GHI'); axes[1].set_xlabel('WSgust (m/s)'); axes[1].set_ylabel('')

sns.scatterplot(x='WD', y='GHI', hue='WS', data=df_clean, alpha=0.7, palette='viridis', ax=axes[2])
axes[2].set_title('Wind Direction vs GHI'); axes[2].set_xlabel('WD (°)'); axes[2].set_ylabel('')
axes[2].legend(title='WS (m/s)', loc='upper right', fontsize=8)

plt.suptitle('Wind Parameters vs Solar Irradiance', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# === 3. HUMIDITY RELATIONSHIPS ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.scatterplot(x='RH', y='Tamb', data=df_clean, alpha=0.6, color='purple', ax=axes[0])
axes[0].set_title('RH vs Ambient Temperature'); axes[0].set_xlabel('RH (%)'); axes[0].set_ylabel('Tamb (°C)')

sns.scatterplot(x='RH', y='GHI', data=df_clean, alpha=0.6, color='red', ax=axes[1])
axes[1].set_title('RH vs GHI'); axes[1].set_xlabel('RH (%)'); axes[1].set_ylabel('GHI (W/m²)')

plt.suptitle('Humidity Impact', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# === 4. KEY CORRELATIONS ===
print("KEY CORRELATIONS:")
print(f"  GHI ↔ DNI   : {df_clean['GHI'].corr(df_clean['DNI']):.3f}")
print(f"  GHI ↔ TModA : {df_clean['GHI'].corr(df_clean['TModA']):.3f}")
print(f"  RH  ↔ GHI   : {df_clean['RH'].corr(df_clean['GHI']):.3f}")
print(f"  WS  ↔ GHI   : {df_clean['WS'].corr(df_clean['GHI']):.3f}")

# Insight: Strong solar correlation, humidity suppresses GHI, wind has weak effect

# %% [markdown]
# ## 6. Wind & Distribution Analysis

# %%
# =============================================
# WIND & DISTRIBUTION ANALYSIS (ONE CELL)
# =============================================
import matplotlib.pyplot as plt
from windrose import WindroseAxes

# === 1. WIND ROSE: WS & WD ===
ax = WindroseAxes.from_ax(figsize=(8, 8))
ax.bar(df_clean['WD'], df_clean['WS'], 
       normed=True, 
       opening=0.8, 
       edgecolor='white',
       cmap=plt.cm.viridis)
ax.set_legend(title='Wind Speed (m/s)', loc='lower left', bbox_to_anchor=(1.1, 0))
ax.set_title('Wind Rose: Direction & Speed Distribution', fontsize=14, pad=20)
plt.show()

# === 2. HISTOGRAMS: GHI + WS ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# GHI Histogram
df_clean['GHI'].hist(bins=60, color='orange', alpha=0.8, edgecolor='black', linewidth=0.5, ax=axes[0])
axes[0].set_title('GHI Distribution', fontsize=14)
axes[0].set_xlabel('GHI (W/m²)')
axes[0].set_ylabel('Frequency')
axes[0].grid(True, alpha=0.3)

# WS Histogram
df_clean['WS'].hist(bins=50, color='skyblue', alpha=0.8, edgecolor='black', linewidth=0.5, ax=axes[1])
axes[1].set_title('Wind Speed Distribution', fontsize=14)
axes[1].set_xlabel('Wind Speed (m/s)')
axes[1].set_ylabel('Frequency')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Solar & Wind Distributions', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# === 3. QUICK STATS ===
print("WIND & GHI DISTRIBUTION SUMMARY:")
print(f"  GHI mean: {df_clean['GHI'].mean():.2f} W/m² | std: {df_clean['GHI'].std():.2f}")
print(f"  WS  mean: {df_clean['WS'].mean():.2f} m/s  | max: {df_clean['WS'].max():.2f}")
print(f"  Dominant wind direction: {df_clean['WD'].mode().iloc[0]:.1f}°")

# %% [markdown]
# ##  7 Temperature & Humidity Analysis
# Examine how **relative humidity (RH)** influences **ambient temperature (Tamb)** and **solar radiation (GHI)**.
# 
# **Goals:**
# - Correlation between RH and Tamb
# - Correlation between RH and GHI
# - Binned RH → average Tamb & GHI
# - Insight: High humidity → clouds → lower GHI, cooler Tamb

# %%
# =============================================
# TEMPERATURE & HUMIDITY ANALYSIS (ONE CELL)
# =============================================
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. CORRELATION VALUES ===
print("CORRELATION SUMMARY:")
print(f"  RH vs Tamb : {df_clean['RH'].corr(df_clean['Tamb']):.3f}")
print(f"  RH vs GHI  : {df_clean['RH'].corr(df_clean['GHI']):.3f}")

# === 2. SCATTER PLOTS: RH vs Tamb & RH vs GHI ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RH vs Tamb
sns.scatterplot(x='RH', y='Tamb', data=df_clean, alpha=0.6, color='purple', ax=axes[0])
axes[0].set_title('Relative Humidity vs Ambient Temperature')
axes[0].set_xlabel('RH (%)')
axes[0].set_ylabel('Tamb (°C)')
axes[0].grid(True, alpha=0.3)

# RH vs GHI
sns.scatterplot(x='RH', y='GHI', data=df_clean, alpha=0.6, color='red', ax=axes[1])
axes[1].set_title('Relative Humidity vs Solar Radiation')
axes[1].set_xlabel('RH (%)')
axes[1].set_ylabel('GHI (W/m²)')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Humidity Impact on Temperature & Solar', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# === 3. BINNED ANALYSIS: RH Levels ===
df_clean['RH_bin'] = pd.cut(df_clean['RH'], 
                            bins=[0, 40, 70, 100], 
                            labels=['Low (0-40%)', 'Medium (40-70%)', 'High (70-100%)'])

rh_effect = df_clean.groupby('RH_bin')[['Tamb', 'GHI']].mean()

rh_effect.plot(kind='bar', figsize=(10, 5), color=['skyblue', 'orange'], alpha=0.8)
plt.title('Average Tamb & GHI by Humidity Level')
plt.ylabel('Average Value')
plt.xlabel('Humidity Level')
plt.xticks(rotation=0)
plt.legend(title='Variable')
plt.grid(True, alpha=0.3)
plt.show()

# === 4. INSIGHT ===
print("\nINSIGHT:")
print("  • High RH → Lower GHI (clouds scatter sunlight)")
print("  • High RH → Slightly lower Tamb (evaporative cooling)")
print("  • Low RH → Peak solar radiation & higher temperatures")

# %% [markdown]
# ## 8 Bubble Chart
# **GHI vs Tamb** with **bubble size = RH** and **color = BP**
# 
# **Insight:**
# - Larger bubbles → higher humidity
# - Color scale → barometric pressure
# - High GHI + high Tamb + low RH = ideal solar conditions

# %%
# =============================================
# BUBBLE CHART: GHI vs Tamb (Size = RH, Color = BP)
# =============================================
import matplotlib.pyplot as plt

# Normalize bubble size for better visuals
bubble_size = df_clean['RH'] * 3  # Scale RH (0–100) → visible bubbles

plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    x=df_clean['Tamb'], 
    y=df_clean['GHI'],
    s=bubble_size,                    # Bubble size = RH
    c=df_clean['BP'],                 # Color = Barometric Pressure
    cmap='plasma', 
    alpha=0.6,
    edgecolors='black', 
    linewidth=0.5
)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Barometric Pressure (hPa)', fontsize=12)

# Labels & Title
plt.xlabel('Ambient Temperature (Tamb) [°C]', fontsize=12)
plt.ylabel('Global Horizontal Irradiance (GHI) [W/m²]', fontsize=12)
plt.title('GHI vs Temperature\nBubble Size = Relative Humidity (RH)', 
          fontsize=16, pad=20)

# Grid
plt.grid(True, alpha=0.3)

# Optional: Add size legend (manual)
for rh_val in [20, 50, 80]:
    plt.scatter([], [], s=rh_val*3, c='gray', alpha=0.6, label=f'{rh_val}% RH')
plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title='Bubble Size', loc='upper left')

plt.tight_layout()
plt.show()

# === INSIGHT ===
print("BUBBLE CHART INSIGHT:")
print("  • Large bubbles (high RH) → lower GHI (clouds)")
print("  • High Tamb + high GHI + small bubble = clear, hot, sunny")
print("  • BP variation minor — stable atmospheric pressure")

# %%


