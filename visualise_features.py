import pandas
import matplotlib.pyplot as plot
from sklearn.preprocessing import MinMaxScaler


def visualise(df):
# --------------------Print Correlations--------------------

    for column in df.columns:
        if column not in ['url', 'label', 'tld']:
            print(f'--- {column} ---')
            print(df.groupby('label')[[column]].mean())
            print()
# --------------------Visualise Correlation--------------------
    numeric_cols = [c for c in df.columns if c not in ['url', 'label', 'tld']]

    correlations = df[numeric_cols].corrwith(df['label']).sort_values(key=abs, ascending=False)

    print(correlations)

    plot.figure(figsize=(8, 10))
    correlations.sort_values().plot(kind='barh', color=['red' if c < 0 else 'green' for c in correlations.sort_values()])
    plot.title('Feature Correlation with Phishing Label')
    plot.xlabel('Correlation Coefficient with Phishing Label (negative = Benign, Positive = Phishing)')
    plot.axvline(0, color='black', linewidth=0.8)
    plot.tight_layout()
    plot.show()

# --------------------Visualise correlation of features with eachother (helps identify potential redundancies)--------------------
    numeric_cols = [c for c in df.columns if c not in ['url', 'tld']]

    corr_matrix = df[numeric_cols].corr()

    plot.figure(figsize=(14, 12))
    plot.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    plot.colorbar(label='Correlation Coefficient')
    plot.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plot.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plot.title('Feature Correlation Matrix')
    plot.tight_layout()
    plot.show()

    

   