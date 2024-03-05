"""Plotting functions"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from .utilities import create_save_path
from .params import COLORS

rcParams["savefig.dpi"] = 600


def calc_pca(df, tag, is_3d=False):
    if is_3d:
        pc3d = PCA(n_components=3).fit_transform(df)
        df_pc3d = pd.DataFrame(data=pc3d, columns=["PC 1", "PC 2", "PC 3"])
        df_final3d = pd.concat([df_pc3d, tag[["Tag"]]], axis=1)
        return df_final3d
    prc = PCA(n_components=2).fit_transform(df)
    df_prc = pd.DataFrame(data=prc, columns=["PC 1", "PC 2"])
    df_prc = pd.concat([df_prc, tag[["Tag"]]], axis=1)
    return df_prc


def calc_umap(df, is_3d=False):
    if is_3d:
        return umap.UMAP(n_components=3).fit_transform(df)
    return umap.UMAP().fit_transform(df)


def calc_tsne(df, is_3d=False):
    if is_3d:
        return TSNE(n_components=3, random_state=0).fit_transform(df.copy())
    return TSNE(n_components=2, random_state=0).fit_transform(df.copy())


def makefig(tag, names=["PCA", "UMAP", "t-SNE"]):
    fig = plt.figure(figsize=(19, 16))
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=10,
            label=tag,
        )
        for tag, color in zip(tag["Tag"].unique(), COLORS)
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=len(tag["Tag"].unique()),
    )

    fig.suptitle("Dimensionality Reduction Visualizations", fontsize=16)
    gs = gridspec.GridSpec(2, 3, height_ratios=[0.35, 1])
    plt.subplots_adjust(hspace=-0.3)  # adjust the space between plots
    plt.tight_layout()

    pltx1 = []
    pltx2 = []
    for i in range(3):
        pltx1.append(fig.add_subplot(gs[0, i]))
        pltx1[i].set_xlabel(f"{names[i]} 1")
        pltx1[i].set_ylabel(f"{names[i]} 2")

        pltx2.append(fig.add_subplot(gs[1, i], projection="3d"))
        pltx2[i].set_xlabel(f"{names[i]} 1")
        pltx2[i].set_ylabel(f"{names[i]} 2")
        #         pltx2[i].set_zlabel(f"{names[i]} 3")
        pltx2[i].view_init(elev=20, azim=30)
        #         zlabel = pltx2[i].zaxis.get_label()
        #         zlabel.set_rotation(45)

        #         pltx2[i].zaxis.set_rotate_label(True)
        pltx2[i].set_zlabel("")

        # Add a text as a substitute for z-label, and you can rotate this text
        pltx2[i].text2D(
            -0.07,
            0.5,
            f"{names[i]} 3",
            transform=pltx2[i].transAxes,
            rotation=90,
            verticalalignment="center",
        )
    return pltx1, pltx2


def subplt(axis, title, targets, dfd, tag, loc=True, is3d=False):
    axis.set_title(title)
    for target, color in zip(targets, COLORS):
        if loc:
            indicesToKeep = dfd["Tag"] == target
            dfd1 = dfd.loc[indicesToKeep, "PC 1"]
            dfd2 = dfd.loc[indicesToKeep, "PC 2"]
            if is3d:
                dfd3 = dfd.loc[indicesToKeep, "PC 3"]
        else:
            indicesToKeep = tag["Tag"] == target
            dfd1 = dfd[indicesToKeep, 0]
            dfd2 = dfd[indicesToKeep, 1]
            if is3d:
                dfd3 = dfd[indicesToKeep, 2]
        if is3d:
            axis.scatter(dfd1, dfd2, dfd3, c=color, edgecolors="gray", alpha=0.77)
        else:
            axis.scatter(dfd1, dfd2, c=color, edgecolors="gray", alpha=0.77)


def plot_dimensionality_reduction(df, tag_vals, sampling_type, results_path):
    tag = df[["anyadr"]]
    tag["anyadr"] = tag["anyadr"].replace(tag_vals)
    tag.rename(columns={"anyadr": "Tag"}, inplace=True)

    df_dropped = df.drop(columns=["anyadr"])

    principalDf = calc_pca(df_dropped, tag)
    umap_data_2D = calc_umap(df_dropped)
    tsne_data_2D = calc_tsne(df_dropped)
    finalDf3D = calc_pca(df_dropped, tag, True)
    umap_data_3D = calc_umap(df_dropped, True)
    tsne_data_3D = calc_tsne(df_dropped, True)

    targets = tag_vals.values()
    ax1, ax2 = makefig(tag, names=["PCA ", "UMAP", "t-SNE"])
    subplt(ax1[0], "PCA 2D", targets, principalDf, tag, loc=True, is3d=False)
    subplt(ax1[1], "UMAP 2D", targets, umap_data_2D, tag, loc=False, is3d=False)
    subplt(ax1[2], "t-SNE 2D", targets, tsne_data_2D, tag, loc=False, is3d=False)
    subplt(ax2[0], "PCA 3D", targets, finalDf3D, tag, loc=True, is3d=True)
    subplt(ax2[1], "UMAP 3D", targets, umap_data_3D, tag, loc=False, is3d=True)
    subplt(ax2[2], "t-SNE 3D", targets, tsne_data_3D, tag, loc=False, is3d=True)
    # ax2[2].zaxis.labelpad = -0.95
    savepath = create_save_path(
        sampling_type, f"dim_reduct_{sampling_type}.png", results_path
    )
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(df, sampling_type, results_path):
    # Compute pairwise correlation of columns,
    # excluding NA/null values.

    corr_matrix = df.corr(method="pearson")

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20, 10))

    # Draw the heatmap with the mask
    # and correct aspect ratio
    sns.heatmap(
        data=corr_matrix,
        cmap="RdPu",
        annot=False,
        square=True,
        linewidths=0.5,
        fmt=".1f",
    )

    savepath = create_save_path(
        sampling_type, f"corr_heatmap_{sampling_type}.png", results_path
    )
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()


def plot_boxplots(
    *,
    metrics,
    classifier_name,
    sampling_type,
    ylabel_text,
    results_path,
):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    bp = ax.boxplot(metrics, patch_artist=True)  # Note: patch_artist=True

    params = {
        "axes.labelsize": 8,
        "font.size": 8,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.usetex": False,
        "figure.figsize": [10, 4],  # Adjusted for better visualization
    }
    rcParams.update(params)

    # ... (rest of your code regarding spines, axes, ticks, and grid)

    # Getting the tab20b colormap
    cmap = plt.cm.get_cmap("tab20b")

    # Generate colors from the colormap
    num_boxes = len(bp["boxes"])
    colors = [cmap(i / num_boxes) for i in range(num_boxes)]

    # Applying colors to the boxplot components
    for i in range(num_boxes):
        box = bp["boxes"][i]
        box.set_facecolor(colors[i])
        box.set_edgecolor("black")

        # Whiskers color
        bp["whiskers"][i * 2].set_color(colors[i])
        bp["whiskers"][i * 2 + 1].set_color(colors[i])
        bp["whiskers"][i * 2].set_linewidth(2)
        bp["whiskers"][i * 2 + 1].set_linewidth(2)

        # Fliers
        bp["fliers"][i].set(
            markerfacecolor=colors[i],
            marker="o",
            alpha=0.75,
            markersize=6,
            markeredgecolor="none",
        )

        # Medians
        bp["medians"][i].set_color("black")
        bp["medians"][i].set_linewidth(2)

        # Caps
        bp["caps"][i * 2].set_color("black")
        bp["caps"][i * 2 + 1].set_color("black")

    # Further plot adjustments...
    plt.title(f"{classifier_name} performance")
    plt.ylabel(ylabel_text)
    plt.xlabel("No. of Borda consensus ranked features")
    # sampling_type, f"corr_heatmap_{sampling_type}.png", results_path
    savepath = create_save_path(
        sampling_type,
        f"box_plots_{classifier_name}_{ylabel_text}_{sampling_type}.png",
        results_path,
    )
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()


def plot_boxplot_metrics(cross_val_data, sampling_type, results_path):
    classifier_name = cross_val_data["classifier_name"]
    for score_name, results in cross_val_data["scores"].items():
        #   metrics, classifier_name, sampling_type, ylabel_text,results_path,
        plot_boxplots(
            metrics=results,
            classifier_name=classifier_name,
            sampling_type=sampling_type,
            ylabel_text=score_name,
            results_path=results_path,
        )


def plot_borda_importance(borda_results, sampling_type, results_path):
    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(
        data=borda_results,
        x="Borda Rank",
        y="Feature",
        s=100,
        color="steelblue",
        label=" ",
    )
    x_start = borda_results["Borda Rank"].min()
    ax.set_xlim(left=x_start)
    ax.set_facecolor("whitesmoke")
    for _, row in borda_results.iterrows():
        ax.plot(
            [x_start, row["Borda Rank"]],
            [row["Feature"], row["Feature"]],
            color="steelblue",
            linewidth=0.5,
        )
    ax.set_title("Borda Consensus Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.legend_.remove()
    plt.tight_layout()
    savepath = create_save_path(
        sampling_type, f"borda_importance_{sampling_type}.png", results_path
    )
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()
