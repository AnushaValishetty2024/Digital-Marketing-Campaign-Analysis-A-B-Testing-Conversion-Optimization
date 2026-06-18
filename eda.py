import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("digital_marketing_campaign_dataset.csv.csv")

print(df.head())
print(df.columns)
print(df.shape)
print(df.columns)
print(df.info())
print(df.isnull().sum())
df.drop_duplicates(inplace=True)
print(df.describe())
#visualisation 1
plt.figure(figsize=(8,5))

sns.countplot(x='CampaignType', data=df)

plt.title("Campaign Type Distribution")

plt.xticks(rotation=45)

plt.savefig("outputs/campaign_distribution.png")

plt.show()
#visualisation3 
plt.figure(figsize=(6,4))

sns.countplot(x='Age_Group', data=df)

plt.title("Age Group Distribution")

plt.savefig("outputs/age_group_distribution.png")

plt.show()
#visualisation 4
campaign_perf = df.groupby('CampaignType')['Conversion'].mean()

plt.figure(figsize=(8,5))

campaign_perf.plot(kind='bar')

plt.title("Average Conversion Rate by Campaign Type")

plt.ylabel("Conversion Rate")

plt.savefig("outputs/campaign_performance.png")

plt.show()
#visualisation 5
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12,8))


sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.savefig("outputs/correlation_heatmap.png")

plt.show()

df.to_csv("outputs/cleaned_data.csv", index=False)
print("Cleaned dataset saved")

print("\n==============================")
print("EDA COMPLETED SUCCESSFULLY")
print("==============================")

print("\nFinal Dataset Shape:", df.shape)
print("\nColumns Used:", df.columns)


plt.close('all')
