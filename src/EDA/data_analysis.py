import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import h5py
from scipy import stats

def analyze_lesion_characteristics(metadata_df):
    """
    Analyze lesion characteristics by malignancy
    """
    # Select relevant features for analysis
    lesion_features = [
        'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean',
        'tbp_lv_deltaLBnorm', 'tbp_lv_norm_border', 'tbp_lv_norm_color'
    ]

    print("Analyzing lesion characteristics...")

    plt.figure(figsize=(20, 15))
    for idx, feature in enumerate(lesion_features, 1):
        plt.subplot(3, 2, idx)
        sns.boxplot(x='target', y=feature, data=metadata_df)
        plt.title(f'{feature} by Malignancy')
        plt.xticks([0, 1], ['Benign', 'Malignant'])
    plt.tight_layout()
    plt.show()

def analyze_demographics(metadata_df):
    """
    Analyze demographic patterns
    """
    print("\nAnalyzing demographic patterns...")

    # Age distribution by malignancy
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='target', y='age_approx', data=metadata_df)
    plt.title('Age Distribution by Malignancy')
    plt.xticks([0, 1], ['Benign', 'Malignant'])

    # Anatomical site distribution
    plt.subplot(1, 2, 2)
    site_mal = pd.crosstab(metadata_df['anatom_site_general'], metadata_df['target'], normalize='columns')
    site_mal.plot(kind='bar', stacked=True)
    plt.title('Anatomical Site Distribution by Malignancy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_image_properties(metadata_df):
    """
    Analyze image properties
    """
    print("\nAnalyzing image properties...")

    # Create scatter plot of lesion area vs border irregularity
    plt.figure(figsize=(10, 6))
    plt.scatter(metadata_df[metadata_df['target']==0]['tbp_lv_areaMM2'],
                metadata_df[metadata_df['target']==0]['tbp_lv_norm_border'],
                alpha=0.5, label='Benign')
    plt.scatter(metadata_df[metadata_df['target']==1]['tbp_lv_areaMM2'],
                metadata_df[metadata_df['target']==1]['tbp_lv_norm_border'],
                alpha=0.5, label='Malignant')
    plt.xlabel('Lesion Area (mm²)')
    plt.ylabel('Border Irregularity')
    plt.title('Lesion Area vs Border Irregularity')
    plt.legend()
    plt.show()

def print_statistical_tests(metadata_df):
    """
    Perform statistical tests on key features
    """
    print("\nStatistical Tests:")
    print("-" * 50)

    features_to_test = [
        'tbp_lv_areaMM2', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
        'age_approx', 'tbp_lv_deltaLBnorm'
    ]

    for feature in features_to_test:
        try:
            benign = metadata_df[metadata_df['target']==0][feature].dropna()
            malignant = metadata_df[metadata_df['target']==1][feature].dropna()

            # Perform Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(benign, malignant, alternative='two-sided')
            print(f"\n{feature}:")
            print(f"Mann-Whitney U test p-value: {p_value:.4e}")

            # Calculate effect size (Cohen's d)
            d = (np.mean(malignant) - np.mean(benign)) / np.sqrt((np.var(malignant) + np.var(benign)) / 2)
            print(f"Effect size (Cohen's d): {d:.4f}")

        except Exception as e:
            print(f"Error analyzing {feature}: {str(e)}")

def main():
    # Load metadata
    metadata_df = pd.read_csv('train-metadata.csv', low_memory=False)

    # Analyze lesion characteristics
    analyze_lesion_characteristics(metadata_df)

    # Analyze demographics
    analyze_demographics(metadata_df)

    # Analyze image properties
    analyze_image_properties(metadata_df)

    # Perform statistical tests
    print_statistical_tests(metadata_df)

if __name__ == "__main__":
    main()
