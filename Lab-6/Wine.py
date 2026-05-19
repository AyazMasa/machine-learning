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

# === Load datasets ===
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path1 = os.path.join(script_dir, "winequality-red.csv")
file_path2 = os.path.join(script_dir, "winequality-white.csv")

red_df = pd.read_csv(file_path1, sep=';')
white_df = pd.read_csv(file_path2, sep=';')

red_df["wine_type"] = "red"
white_df["wine_type"] = "white"
wine_df = pd.concat([red_df, white_df], axis=0).reset_index(drop=True)

wine_df['quality_category'] = wine_df['quality'].apply(lambda x: 'low' if x <= 5 else 'medium' if x == 6 else 'high')

# === Plot functions ===

def plot_univariate():
    wine_df.hist(figsize=(14, 10), bins=20)
    plt.tight_layout()
    plt.savefig('figures/univariate_analysis.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_quality_distribution():
    sns.countplot(data=wine_df, x='quality')
    plt.title("Wine Quality Distribution")
    plt.savefig('figures/quality_distribution.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_pairwise():
    sns.pairplot(wine_df.sample(300), hue='wine_type', diag_kind='kde')
    plt.savefig('figures/pairplot.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_parallel_coords():
    sampled_df = wine_df.sample(300).copy()
    parallel_coordinates(sampled_df[['alcohol', 'density', 'pH', 'sulphates', 'quality_category']], 
                         'quality_category', color=('#FFE888', '#FF9999', '#88FF88'))
    plt.title("Parallel Coordinates by Quality Category")
    plt.xticks(rotation=45)
    plt.savefig('figures/parallel_coords.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_two_continuous():
    sns.scatterplot(data=wine_df, x='alcohol', y='density', hue='wine_type')
    plt.title("Alcohol vs. Density")
    plt.savefig('figures/alcohol_vs_density.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_joint():
    g = sns.jointplot(data=wine_df, x='alcohol', y='quality', kind='scatter')
    g.figure.savefig('figures/jointplot.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_mixed_attributes():
    sns.boxplot(data=wine_df, x='quality', y='alcohol', hue='wine_type')
    plt.title("Boxplot of Alcohol by Quality and Wine Type")
    plt.savefig('figures/boxplot_mixed.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_multiple_histograms():
    wine_df[['alcohol', 'pH', 'density']].hist(bins=20, figsize=(12, 6))
    plt.tight_layout()
    plt.savefig('figures/multiple_histograms.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_box_violin():
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=wine_df, x='quality', y='alcohol', ax=axs[0])
    sns.violinplot(data=wine_df, x='quality', y='alcohol', ax=axs[1])
    axs[0].set_title("Boxplot")
    axs[1].set_title("Violin Plot")
    plt.tight_layout()
    plt.savefig('figures/box_violin.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_3d_scatter():
    fig = px.scatter_3d(wine_df, x='alcohol', y='density', z='pH', color='quality_category')
    fig.show()  # Not saved
    del fig
    time.sleep(0.5)

def plot_facetgrid():
    g = sns.FacetGrid(wine_df, col='quality_category', hue='wine_type')
    g.map(sns.histplot, 'alcohol', bins=15)
    g.figure.savefig('figures/facetgrid_quality.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_three_discrete():
    wine_df['alcohol_bin'] = pd.cut(wine_df['alcohol'], bins=3, labels=['Low', 'Medium', 'High'])
    wine_df['sulphates_bin'] = pd.cut(wine_df['sulphates'], bins=3, labels=['Low', 'Medium', 'High'])
    g = sns.catplot(x='alcohol_bin', hue='sulphates_bin', col='quality_category', kind='count', data=wine_df)
    g.figure.savefig('figures/three_discrete.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_mixed_swarm():
    sns.swarmplot(data=wine_df, x='quality', y='alcohol', hue='wine_type')
    plt.title("Swarmplot of Alcohol by Quality and Wine Type")
    plt.savefig('figures/swarmplot.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_kde():
    plt.figure(figsize=(10, 6))
    for quality in sorted(wine_df['quality'].unique()):
        subset = wine_df[wine_df['quality'] == quality]
        sns.kdeplot(subset['alcohol'], label=f'Quality {quality}', fill=True)
    plt.title("KDE of Alcohol by Wine Quality")
    plt.legend()
    plt.savefig('figures/kde_plot.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_four_dimensions():
    fig = px.scatter(wine_df, x='alcohol', y='density', color='quality_category',
                     size='sulphates', symbol='wine_type', hover_data=['pH'])
    fig.show()  # Not saved
    del fig
    time.sleep(0.5)

# === FacetGrid examples ===

def facetgrid_plot1():
    g = sns.FacetGrid(wine_df, col="wine_type", hue='quality_category',
                      col_order=['red', 'white'], hue_order=['low', 'medium', 'high'],
                      aspect=1.2, palette=sns.light_palette('navy', 4)[1:])
    g.map(plt.scatter, "volatile acidity", "alcohol", alpha=0.9,
          edgecolor='white', linewidth=0.5, s=100)
    fig = g.figure
    fig.subplots_adjust(top=0.8, wspace=0.3)
    fig.suptitle('Wine Type - Alcohol - Quality - Acidity', fontsize=14)
    fig.savefig('figures/facetgrid1.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def facetgrid_plot2():
    g = sns.FacetGrid(wine_df, col="wine_type", hue='quality_category',
                      col_order=['red', 'white'], hue_order=['low', 'medium', 'high'],
                      aspect=1.2, palette=sns.light_palette('green', 4)[1:])
    g.map(plt.scatter, "volatile acidity", "total sulfur dioxide", alpha=0.9,
          edgecolor='white', linewidth=0.5, s=100)
    fig = g.figure
    fig.subplots_adjust(top=0.8, wspace=0.3)
    fig.suptitle('Wine Type - Sulfur Dioxide - Acidity - Quality', fontsize=14)
    fig.savefig('figures/facetgrid2.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def facetgrid_plot3():
    g = sns.FacetGrid(wine_df, col="wine_type", hue='quality_category',
                      col_order=['red', 'white'], hue_order=['low', 'medium', 'high'],
                      aspect=1.2, palette=sns.light_palette('black', 4)[1:])
    g.map(plt.scatter, "residual sugar", "alcohol", alpha=0.8,
          edgecolor='white', linewidth=0.5)
    fig = g.figure
    fig.subplots_adjust(top=0.8, wspace=0.3)
    fig.suptitle('Wine Type - Residual Sugar - Alcohol - Quality', fontsize=14)
    fig.savefig('figures/facetgrid3.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def facetgrid_plot4():
    g = sns.FacetGrid(wine_df, row='wine_type', col="quality", hue='quality_category')
    g.map(plt.scatter, "residual sugar", "alcohol", alpha=0.5,
          edgecolor='k', linewidth=0.5)
    fig = g.figure
    fig.set_size_inches(18, 8)
    fig.subplots_adjust(top=0.85, wspace=0.3)
    fig.suptitle('Wine Type - Residual Sugar - Alcohol - Quality Class - Quality Rating', fontsize=14)
    fig.savefig('figures/facetgrid4.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

# === Run All Plots ===
plot_univariate()
plot_quality_distribution()
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
facetgrid_plot1()
facetgrid_plot2()
facetgrid_plot3()
facetgrid_plot4()

#  Done!
print(f"\n All 19 plots displayed. 17 saved to 'figures'.")
print(f"⏱️ Total time: {round(time.time() - start, 2)} seconds.")
