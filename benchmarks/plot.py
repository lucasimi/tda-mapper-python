import math

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator, NullLocator


alpha = 0.75


df_names = pd.DataFrame({
    'bench': [
        'tda-mapper',
        'tda-mapper-cubical',
        'tda-mapper-ball',
        'giotto-tda',
        'kepler-mapper',
    ],
    'name': [
        'tda-mapper',
        'tda-mapper (cubical cover)',
        'tda-mapper (ball cover)',
        'giotto-tda',
        'kepler-mapper',
    ],
})


df_colors = pd.DataFrame({
    'bench': [
        'tda-mapper',
        'tda-mapper-cubical',
        'tda-mapper-ball',
        'giotto-tda',
        'kepler-mapper',
    ],
    'color': [
        'tab:blue',
        'tab:blue',
        'tab:blue',
        'tab:orange',
        'tab:green',
    ],
})


df_styles = pd.DataFrame({
    'bench': [
        'tda-mapper',
        'tda-mapper-cubical',
        'tda-mapper-ball',
        'giotto-tda',
        'kepler-mapper',
    ],
    'style': [
        '-',
        '-',
        '--',
        '--',
        '--',
    ],
})


df_markers = pd.DataFrame({
    'bench': [
        'tda-mapper',
        'tda-mapper-cubical',
        'tda-mapper-ball',
        'giotto-tda',
        'kepler-mapper',
    ],
    'marker': [
        'D',
        'D',
        'D',
        's',
        'o',
    ],
})


df_zorder = pd.DataFrame({
    'bench': [
        'tda-mapper',
        'tda-mapper-cubical',
        'tda-mapper-ball',
        'giotto-tda',
        'kepler-mapper',
    ],
    'zorder': [
        5,
        4,
        3,
        1,
        2,
    ],
})


df_titles = pd.DataFrame({
    'dataset': [
        'line',
        'digits',
        'mnist',
        'cifar10',
        'fashion_mnist',
        'digits_pca',
        'mnist_pca',
        'cifar10_pca',
        'fashion_mnist_pca',
        'digits_umap',
        'mnist_umap',
        'cifar10_umap',
        'fashion_mnist_umap',
    ],
    'title': [
        'Line',
        'Digits',
        'MNIST',
        'Cifar-10',
        'Fashion-MNIST',
        'Digits (PCA)',
        'MNIST (PCA)',
        'Cifar-10 (PCA)',
        'Fashion-MNIST (PCA)',
        'Digits (UMAP)',
        'MNIST (UMAP)',
        'Cifar-10 (UMAP)',
        'Fashion-MNIST (UMAP)',
    ],
})


def load_benchmark(path):
    df = pd.read_csv(path)
    df = pd.merge(
        left=df,
        right=df_names,
        on='bench'
    )
    df = pd.merge(
        left=df,
        right=df_colors,
        on='bench'
    )
    df = pd.merge(
        left=df,
        right=df_styles,
        on='bench'
    )
    df = pd.merge(
        left=df,
        right=df_markers,
        on='bench'
    )
    df = pd.merge(
        left=df,
        right=df_zorder,
        on='bench'
    )
    df = pd.merge(
        left=df,
        right=df_titles,
        on='dataset'
    )
    df.sort_values(by=['dataset', 'bench', 'p', 'k'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_inset(ax):
    inset_ax = ax.inset_axes([0.0, 0.63, 0.37, 0.37])
    inset_ax.set_yscale('log')
    #inset_ax.yaxis.set_label_position('left')
    #inset_ax.tick_params(axis='y', direction='out')
    #inset_ax.tick_params(axis='x', direction='out')
    #inset_ax.yaxis.tick_left()
    #inset_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    inset_ax.xaxis.set_major_locator(NullLocator())
    inset_ax.xaxis.set_minor_locator(NullLocator())
    #inset_ax.yaxis.set_major_locator(LogLocator(base=10.0))
    inset_ax.yaxis.set_major_locator(NullLocator())
    inset_ax.yaxis.set_minor_locator(NullLocator())
    for _, spine in inset_ax.spines.items():
        spine.set_alpha(alpha)
    inset_ax.xaxis.label.set_alpha(alpha)
    inset_ax.yaxis.label.set_alpha(alpha)
    for tickline in inset_ax.get_xticklines():
        tickline.set_alpha(alpha)
    for label in inset_ax.get_xticklabels():
        label.set_alpha(alpha)
        label.set_fontsize(8)
    for tickline in inset_ax.get_yticklines():
        tickline.set_alpha(alpha)
    for label in inset_ax.get_yticklabels():
        label.set_alpha(alpha)
        label.set_fontsize(8)
    return inset_ax


def plot_library(df_bench, ax, ax_log):
    assert len(df_bench.bench.unique()) == 1
    assert len(df_bench.name.unique()) == 1
    assert len(df_bench.color.unique()) == 1
    assert len(df_bench['style'].unique()) == 1
    assert len(df_bench.marker.unique()) == 1
    assert len(df_bench.zorder.unique()) == 1
    #bench = df_bench.bench.values[0]
    name = df_bench.name.values[0]
    color = df_bench.color.values[0]
    style = df_bench['style'].values[0]
    marker = df_bench.marker.values[0]
    zorder = df_bench.zorder.values[0]
    df_plot = df_bench.sort_values(by='k')
    if ax_log is not None:
        ax_log.plot(
            df_plot.k,
            df_plot.time,
            label=name,
            color=color,
            linestyle=style,
            marker=marker,
            markersize=2,
            markerfacecolor='white',
            markeredgecolor=color,
            markeredgewidth=1.0,
            alpha=alpha,
            linewidth=1.0,
            zorder=zorder,
        )
    line = ax.plot(
        df_plot.k,
        df_plot.time,
        label=name,
        color=color,
        linestyle=style,
        marker=marker,
        markersize=4,
        markerfacecolor='white',
        markeredgecolor=color,
        markeredgewidth=1.5,
        alpha=1.0,
        linewidth=1.5,
        zorder=zorder,
    )[0]
    return line


def plot_benchmark(df_p, ax, ax_log):
    assert len(df_p.p.unique()) == 1
    p = df_p.p.values[0]
    max_time = df_p.time.max()
    min_time = df_p.time.min()
    ax.set_title(f'p = {p}')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()
    for bench in df_p.bench.unique():
        df_bench = df_p[df_p.bench == bench]
        plot_library(df_bench, ax, ax_log)


def plot_experiment(df_dataset, fig, axes, axes_log):
    for j, p in enumerate(df_dataset.p.unique()):
        df_p = df_dataset[df_dataset.p == p]
        plot_benchmark(df_p, axes[j], axes_log[j])
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    unique = dict(zip(labels, lines))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc='upper center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
    )


def plot_split(df):
    for dataset in df.dataset.unique():
        df_dataset = df[df.dataset == dataset]
        assert len(df_dataset.title.unique()) == 1
        p_count = df_dataset.p.nunique()
        fig, axes = plt.subplots(
            nrows=1,
            ncols=p_count,
            figsize=(8, 3.2),
            dpi=300,
            sharex=False,
            sharey=False,
        )
        axes_log = []
        for ax in axes:
            ax.set_xlabel('k')
            ax.set_ylabel('time (s)')
            inset_ax = get_inset(ax)
            #inset_ax = None
            axes_log.append(inset_ax)
        plot_experiment(df_dataset, fig, axes, axes_log)
        title = df_dataset.title.values[0]
        plt.suptitle(f'{title}', fontsize=14)
        plt.tight_layout()
        fig.savefig(f'benchmark_{dataset}.png', bbox_inches='tight')
        #plt.show()


def plot_full(df):
    p_count = df.p.nunique()
    fig, axes = plt.subplots(
        nrows=3,
        ncols=p_count,
        figsize=(8, 8),
        dpi=300,
        sharex=False,
        sharey=False,
    )
    for i, dataset in enumerate(df.dataset.unique()):
        df_dataset = df[df.dataset == dataset]
        p_count = df_dataset.p.nunique()
        for j, p in enumerate(df_dataset.p.unique()):
            df_p = df_dataset[df_dataset.p == p]
            title = df_titles[df_titles.dataset == dataset].title.values[0]
            axes[i, j].set_title(f'{title}, p = {p}')
            axes[i, j].set_xlabel('k')
            axes[i, j].set_ylabel('time (s)')
            axes[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))
            for bench in df_p.bench.unique():
                df_bench = df_p[df_p.bench == bench]
                assert len(df_bench.color.unique()) == 1
                assert len(df_bench['style'].unique()) == 1
                assert len(df_bench.marker.unique()) == 1
                assert len(df_bench.zorder.unique()) == 1
                color = df_bench.color.values[0]
                style = df_bench['style'].values[0]
                marker = df_bench.marker.values[0]
                zorder = df_bench.zorder.values[0]
                axes[i, j].plot(
                    df_bench.k,
                    df_bench.time,
                    label=bench,
                    color=color,
                    linestyle=style,
                    marker=marker,
                    markersize=4,
                    markerfacecolor='white',
                    markeredgecolor=color,
                    markeredgewidth=1.5,
                    alpha=1.0,
                    linewidth=1.5,
                    zorder=zorder,
                )
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    unique = dict(zip(labels, lines))
    unique_size = len(unique.keys())
    fig.legend(
        unique.values(),
        unique.keys(),
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=unique_size,
    )
    plt.tight_layout()
    fig.savefig(f'benchmark_{unique_size}.png', bbox_inches='tight')
    #plt.show()


def get_latex_table(df):
    df_approx = df.copy()
    df_approx['time'] = df_approx['time'].round(2)
    df_pivot = pd.pivot_table(
        df_approx,
        values='time',
        index=['k'],
        columns=['dataset', 'p', 'name'],
    )
    return df_pivot.to_latex(float_format="%.2f", na_rep='NaN')


def print_latex_tables(df):
    for dataset in df['dataset'].unique():
        df_dataset_tm = df[
            (df['dataset'] == dataset) &
            (df['bench'].isin([
                'tda-mapper',
                'tda-mapper-cubical',
                'tda-mapper-ball',
            ]))
        ]
        df_dataset_oth = df[
            (df['dataset'] == dataset) &
            ~(df['bench'].isin([
                'tda-mapper',
                'tda-mapper-cubical',
                'tda-mapper-ball',
            ]))
        ]
        df_table_oth = get_latex_table(df_dataset_oth)
        df_table_tm = get_latex_table(df_dataset_tm)
        print(df_table_oth)
        print(df_table_tm)
        print()
        print('====================')
        print()


if __name__ == '__main__':
    files = [
        'benchmark_digits_pca.csv',
        'benchmark_mnist_pca.csv',
        'benchmark_cifar10_pca.csv',
        'benchmark_fmnist_pca.csv',
    ]

    plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})
    for file in files:
        df_benchmark = load_benchmark(file)
        print_latex_tables(df_benchmark)
        plot_split(df_benchmark)

