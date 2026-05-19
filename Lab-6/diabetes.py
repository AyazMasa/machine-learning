import pandas as pd
import os
import time
from matplotlib import pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
import plotly.express as px
import numpy as np

# Start timing
start = time.time()

# === Setup: Handle 'figures' directory ===
figures_dir = "figures"
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
else:
    for f in os.listdir(figures_dir):
        os.remove(os.path.join(figures_dir, f))

# === Load diabetes dataset ===
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "diabetes.csv")
df = pd.read_csv(file_path)
df["OutcomeLabel"] = df["Outcome"].map({0: "No Diabetes", 1: "Diabetes"})

# === Plot functions ===
def plot_univariate():
    df.hist(figsize=(14, 10), bins=20)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/univariate_analysis.png', dpi=300)
    plt.show()
    plt.close()

def plot_outcome_distribution():
    sns.countplot(data=df, x='OutcomeLabel')
    plt.title("Diabetes Outcome Distribution")
    plt.savefig(f'{figures_dir}/outcome_distribution.png', dpi=300)
    plt.show()
    plt.close()

def plot_pairwise():
    sns.pairplot(df.sample(300), hue='OutcomeLabel', diag_kind='kde')
    plt.savefig(f'{figures_dir}/pairplot.png', dpi=300)
    plt.show()
    plt.close()

def plot_parallel_coords():
    sampled_df = df.sample(300).copy()
    parallel_coordinates(sampled_df[['Glucose', 'BMI', 'Age', 'BloodPressure', 'OutcomeLabel']], 
                         'OutcomeLabel')
    plt.title("Parallel Coordinates by Outcome")
    plt.xticks(rotation=45)
    plt.savefig(f'{figures_dir}/parallel_coords.png', dpi=300)
    plt.show()
    plt.close()

def plot_two_continuous():
    sns.scatterplot(data=df, x='Glucose', y='BMI', hue='OutcomeLabel')
    plt.title("Glucose vs. BMI")
    plt.savefig(f'{figures_dir}/glucose_vs_bmi.png', dpi=300)
    plt.show()
    plt.close()

def plot_joint():
    g = sns.jointplot(data=df, x='Glucose', y='Age', kind='scatter')
    g.figure.savefig(f'{figures_dir}/jointplot.png', dpi=300)
    plt.show()
    plt.close()

def plot_mixed_attributes():
    sns.boxplot(data=df, x='OutcomeLabel', y='Glucose')
    plt.title("Boxplot of Glucose by Outcome")
    plt.savefig(f'{figures_dir}/boxplot_mixed.png', dpi=300)
    plt.show()
    plt.close()

def plot_multiple_histograms():
    df[['Glucose', 'BMI', 'Age']].hist(bins=20, figsize=(12, 6))
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/multiple_histograms.png', dpi=300)
    plt.show()
    plt.close()

def plot_box_violin():
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x='OutcomeLabel', y='Glucose', ax=axs[0])
    sns.violinplot(data=df, x='OutcomeLabel', y='Glucose', ax=axs[1])
    axs[0].set_title("Boxplot")
    axs[1].set_title("Violin Plot")
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/box_violin.png', dpi=300)
    plt.show()
    plt.close()

def plot_3d_scatter():
    fig = px.scatter_3d(df, x='Glucose', y='BMI', z='Age', color='OutcomeLabel')
    fig.show()

def plot_facetgrid():
    g = sns.FacetGrid(df, col='OutcomeLabel')
    g.map(sns.histplot, 'Glucose', bins=15)
    g.figure.savefig(f'{figures_dir}/facetgrid_outcome.png', dpi=300)
    plt.show()
    plt.close()

def plot_three_discrete():
    df['Glucose_bin'] = pd.cut(df['Glucose'], bins=3, labels=['Low', 'Medium', 'High'])
    df['BMI_bin'] = pd.cut(df['BMI'], bins=3, labels=['Low', 'Medium', 'High'])
    g = sns.catplot(x='Glucose_bin', hue='BMI_bin', col='OutcomeLabel', kind='count', data=df)
    g.figure.savefig(f'{figures_dir}/three_discrete.png', dpi=300)
    plt.show()
    plt.close()

def plot_mixed_swarm():
    sns.swarmplot(data=df, x='OutcomeLabel', y='Glucose')
    plt.title("Swarmplot of Glucose by Outcome")
    plt.savefig(f'{figures_dir}/swarmplot.png', dpi=300)
    plt.show()
    plt.close()

def plot_kde():
    plt.figure(figsize=(10, 6))
    for outcome in [0, 1]:
        subset = df[df['Outcome'] == outcome]
        sns.kdeplot(subset['Glucose'], label=f'Outcome {outcome}', fill=True)
    plt.title("KDE of Glucose by Outcome")
    plt.legend()
    plt.savefig(f'{figures_dir}/kde_plot.png', dpi=300)
    plt.show()
    plt.close()

def plot_four_dimensions():
    fig = px.scatter(df, x='Glucose', y='BMI', color='OutcomeLabel',
                     size='Age', symbol='Outcome', hover_data=['Insulin'])
    fig.show()

# === Run All Plots ===
plot_univariate()
plot_outcome_distribution()
plot_pairwise()
plot_parallel_coords()
plot_two_continuous()
plot_joint()
plot_mixed_attributes()
plot_multiple_histograms()
plot_box_violin()
plot_3d_scatter()
plot_facetgrid()
plot_three_discrete()
plot_mixed_swarm()
plot_kde()
plot_four_dimensions()

print("\n All 15 plots (13 saved) displayed successfully.")
print(f"⏱️ Total time: {round(time.time() - start, 2)} seconds.")
