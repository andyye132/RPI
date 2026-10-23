from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, LogFormatterMathtext, NullLocator
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

NAMES = ('experiment_a_compact', 'interface_errors', 'translation_directions')
METHODS = {'Joint': '^', 'RPI': 'o', 'Per-step map': 'D', 'Scalar': 's',
           'RPI (fixed map)': 'D', 'RPI (uncertain map)': 'o'}
PALETTE = sns.color_palette('colorblind')
COLORS = {'Joint': PALETTE[0], 'RPI': PALETTE[2], 'Per-step map': PALETTE[1], 'Scalar': '#707070',
          'RPI (fixed map)': PALETTE[1], 'RPI (uncertain map)': PALETTE[2]}
MAPS = ('constant', 'x_only', 'two_dimensional')
C_METHODS = ('scalar', 'rpi', 'rpi_uncertain')
C_LABELS = ('Scalar', 'RPI (fixed map)', 'RPI (uncertain map)')
A_METHODS = ('joint', 'corrected', 'naive_slotwise')
B_METHODS = ('scalar', 'message', 'joint', 'rpi_uncertain')
B_LABELS = {'scalar': 'Scalar', 'message': 'RPI (fixed map)', 'joint': 'Joint',
            'rpi_uncertain': 'RPI (uncertain map)'}
LABELS = {'joint': 'Joint', 'corrected': 'RPI', 'naive_slotwise': 'Per-step map',
          'scalar': 'Scalar', 'message': 'RPI'}
SCENARIOS = ('correct', 'registration', 'attenuation', 'combined')
B_METRICS = ('common_rmse_final_third', 'relative_rmse_final_third',
             'terminal_common_rms_uncertainty')


class IntervalSymbol:
    pass


class IntervalHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xd, yd, width, height, fontsize, trans):
        x0, x1, middle = -xd, -xd+width, -yd+height/2
        return [Line2D([x0, x1], [middle, middle], color='#353535', lw=1.3, transform=trans),
                Line2D([x0, x0], [middle-height*.35, middle+height*.35], color='#353535', lw=1.15, transform=trans),
                Line2D([x1, x1], [middle-height*.35, middle+height*.35], color='#353535', lw=1.15, transform=trans)]


def ci(values: np.ndarray, axis: int = 0) -> dict:
    values = np.asarray(values)
    n = values.shape[axis]
    mean = values.mean(axis=axis)
    sd = values.std(axis=axis, ddof=1)
    half = stats.t.ppf(.975, n - 1) * sd / np.sqrt(n)
    return {"mean": np.asarray(mean).tolist(), "sd": np.asarray(sd).tolist(),
            "ci95": np.stack([mean - half, mean + half], axis=-1).tolist()}


def load_a(path):
    trials = pd.read_csv(path/'replicates.csv')
    metrics = {'normalized_full_trajectory_nees': 'normalized_nees',
               'normalized_terminal_common_nees': 'terminal_common_normalized_nees'}
    result = {}
    with np.load(path/'raw_draws_and_posteriors.npz', allow_pickle=False) as raw:
        joint_cov = raw['joint_posterior_covariance']
        for method in A_METHODS:
            rows = trials[trials.method == method]
            result[method] = {metric: ci(rows[column].to_numpy()) for metric, column in metrics.items()}
            result[method]['covariance_difference'] = float(
                np.linalg.norm(raw[f'{method}_posterior_covariance']-joint_cov)/np.linalg.norm(joint_cov))
    return result


def load_b(path):
    seeds = json.loads((path/'protocol.json').read_text())['seeds']
    values = {(scenario, method, metric): [] for scenario in SCENARIOS
              for method in B_METHODS for metric in B_METRICS}
    for seed in seeds:
        with np.load(path/f'seed_{seed}.npz', allow_pickle=False) as raw:
            truth = raw['truth']
            steps, robots, _ = truth.shape
            centroid = np.tile(np.eye(2)/robots, (1, robots))
            for scenario in SCENARIOS:
                for method in B_METHODS:
                    error = raw[f'{scenario}_{method}_estimate']-truth
                    common = error.mean(axis=1)
                    relative = error-common[:, None, :]
                    covariance = raw[f'{scenario}_{method}_terminal_covariance']
                    computed = {
                        'common_rmse_final_third': np.sqrt(np.mean(np.sum(common[2*steps//3:]**2, axis=-1))),
                        'relative_rmse_final_third': np.sqrt(np.mean(np.sum(relative[2*steps//3:]**2, axis=-1))),
                        'terminal_common_rms_uncertainty': np.sqrt(np.trace(centroid@covariance@centroid.T)),
                    }
                    for metric, value in computed.items():
                        values[(scenario, method, metric)].append(float(value))
    return {key: ci(np.asarray(trials)) for key, trials in values.items()}


def load_c(path):
    seeds = json.loads((path/'summary.json').read_text())['protocol']['seeds']
    values = {(condition, method, axis): [] for condition in MAPS
              for method in C_METHODS for axis in ('x', 'y')}
    for seed in seeds:
        with np.load(path/f'seed_{seed}.npz', allow_pickle=False) as raw:
            for condition in MAPS:
                for method in C_METHODS:
                    common = (raw[f'{condition}_{method}_estimate']-raw['truth']).mean(axis=1)
                    for i, axis in enumerate(('x', 'y')):
                        value = float(np.sqrt(np.mean(common[2*len(common)//3:, i]**2)))
                        values[(condition, method, axis)].append(value)
    return {key: ci(np.asarray(trials)) for key, trials in values.items()}


def load_noise(path):
    levels = sorted(map(float, json.loads((path/'summary.json').read_text())['levels']))
    rows = []
    for sigma in levels:
        with np.load(path/f'sigma_{sigma:g}'/'raw_draws_and_posteriors.npz', allow_pickle=False) as raw:
            for method in A_METHODS:
                error = (raw[f'{method}_posterior_mean']-raw['true_pose_perturbations'])[-6:]
                values = np.linalg.norm(error.reshape(3, 2, -1).mean(axis=0), axis=0)
                rows.append({'sigma_db': sigma, 'method': LABELS[method], **ci(values)})
    return rows


def set_style():
    family = 'sans-serif'
    sns.set_theme(context='paper', style='whitegrid', font=family, rc={
        'font.sans-serif': ['Helvetica', 'Arial', 'Liberation Sans', 'DejaVu Sans'],
        'font.family': family, 'font.size': 10.5, 'axes.labelsize': 10.5,
        'axes.titlesize': 10.8, 'axes.titleweight': 'normal', 'axes.linewidth': .8,
        'xtick.labelsize': 10.0, 'ytick.labelsize': 10.0, 'legend.fontsize': 10.0,
        'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black',
        'ytick.color': 'black', 'grid.linewidth': .6, 'grid.alpha': .5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
        'mathtext.fontset': 'dejavusans',
    })
    return family


def canvas(n, width, height, *, noise=False):
    fig = plt.figure(figsize=(width, height))
    geometry = ((.70, 1.94), (2.87, 1.94), (5.04, 1.94)) if n == 3 else ((.76, 2.88), (4.00, 2.98))
    if noise:
        geometry = ((.56, 2.57), (3.55, 1.51), (5.42, 1.56))
    axes = [fig.add_axes([left/width, .46/height, plot_width/width, (height-1.02)/height])
            for left, plot_width in geometry]
    for ax in axes:
        ax.grid(axis='y', visible=False)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', length=3, width=.75, pad=4, labelsize=9.0)
        ax.tick_params(axis='y', length=0, pad=5, labelsize=9.2)
        sns.despine(ax=ax, left=True)
        ax.xaxis.label.set_fontsize(9.5)
        ax._direct_labels = []
    return fig, axes


def add_interval(ax, y, stat, method, digits=3, covariance=False):
    mean = float(stat['mean'])
    lo, hi = (mean, mean) if covariance else map(float, stat['ci95'])
    if not covariance:
        ax.errorbar([mean], [y], xerr=[[max(0., mean-lo)], [max(0., hi-mean)]],
                    fmt='none', ecolor='#353535', elinewidth=1.3,
                    capsize=3.5, capthick=1.15, zorder=3)
    sns.scatterplot(x=[mean], y=[y], marker=METHODS[method],
                    color='white' if method == 'Joint' else COLORS[method],
                    edgecolor='black', linewidth=.8, s=44 if method != 'Joint' else 52,
                    legend=False, ax=ax, zorder=5)
    value_label = f'{mean:.2e}'.replace('e-0', 'e−').replace('e-', 'e−') if covariance and mean < 1e-3 else f'{mean:.{digits}f}'
    text = ax.annotate(value_label, xy=(hi, y), xytext=(6, 0),
                       textcoords='offset points', ha='left', va='center',
                       fontsize=9.2, color='black', annotation_clip=False)
    ax._direct_labels.append((text, y, lo, hi))


def legend(fig, methods, joint_label='Joint'):
    handles = [Line2D([], [], linestyle='none', marker=METHODS[m], markersize=6.5,
                      markerfacecolor='white' if m == 'Joint' else COLORS[m],
                      markeredgecolor='black', markeredgewidth=.8,
                      label=m) for m in methods]
    handles.append(IntervalSymbol())
    display_labels = [joint_label if method == 'Joint' else method for method in methods]
    return fig.legend(handles=handles, labels=display_labels+['95% CI'],
                      handler_map={IntervalSymbol: IntervalHandler()}, loc='upper center', bbox_to_anchor=(.5, .998),
                      frameon=False, borderaxespad=0, handlelength=1.1,
                      ncol=2 if fig.get_figwidth() < 4 and len(handles) > 3 else len(handles),
                      handletextpad=.5, columnspacing=1.1, labelspacing=.65, fontsize=9.5)


def plot_noise(ax, rows):
    for method in ('Joint', 'RPI', 'Per-step map'):
        data = [r for r in rows if r['method'] == method]
        x = np.array([r['sigma_db'] for r in data])
        means = np.array([r['mean'] for r in data])
        limits = np.array([r['ci95'] for r in data])
        sns.lineplot(x=x, y=means, color=COLORS[method], estimator=None, sort=False,
                     linestyle='--' if method == 'Joint' else '-', linewidth=1.5,
                     legend=False, ax=ax, zorder=2)
        if method != 'Joint':
            ax.errorbar(x, means, yerr=np.array([means-limits[:, 0], limits[:, 1]-means]),
                        fmt='none', ecolor='#353535', elinewidth=1.15, capsize=3, capthick=1, zorder=3)
        sns.scatterplot(x=x, y=means, marker=METHODS[method],
                        color='white' if method == 'Joint' else COLORS[method], edgecolor='black',
                        linewidth=.8, s={'Joint': 65, 'RPI': 16, 'Per-step map': 32}[method],
                        legend=False, ax=ax, zorder=4 if method == 'Joint' else 5)
        for r in data:
            value, (lo, hi), sigma = r['mean'], r['ci95'], r['sigma_db']
            label = f'{value:.3f}'
            if method != 'Joint':
                above = method == 'Per-step map'
                text = ax.annotate(label, xy=(sigma, hi if above else lo),
                                   xytext=(0, 6 if above else -6), textcoords='offset points',
                                   ha='center', va='bottom' if above else 'top', fontsize=8.6)
    ax.set(xscale='log', xlim=(.021, 2.74), ylim=(.60, .965),
           xlabel=r'Radio Noise $\sigma$ (dB; log scale)', ylabel='Mean error (m)',
           xticks=sorted({r['sigma_db'] for r in rows}), yticks=[.6, .7, .8, .9])
    ax.set_xticklabels([f'{s:g}' for s in sorted({r['sigma_db'] for r in rows})], fontsize=8.5)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.label.set_fontsize(9.5)
    ax.grid(axis='y', visible=True)
    ax.set_title('(a) Noise sensitivity', pad=7, fontsize=9.8)


def plot_a(a, out, noise):
    fig, ax = canvas(3, 7.16, 2.6, noise=True)
    plot_noise(ax[0], noise)
    for group, key in enumerate(('normalized_full_trajectory_nees', 'normalized_terminal_common_nees')):
        top = 9.4 if group == 0 else 4.4
        ax[1].text(.02, top, ('Trajectory (72D)', 'Centroid (2D)')[group],
                   transform=ax[1].get_yaxis_transform(), ha='left', va='center', fontsize=8.8)
        for i, method_key in enumerate(A_METHODS):
            add_interval(ax[1], top-1.2-i*.95, a[method_key][key], LABELS[method_key])
    ax[1].set_title('(b) Consistency', pad=7, fontsize=9.8)
    ax[1].set(xlabel='NEES / dimension', xlim=(0, 2.7), ylim=(0, 10), yticks=[], xticks=[0, 1, 2])
    for low, high in ((5.75, 8.65), (.75, 3.65)):
        ax[1].plot([1, 1], [low, high], color='#555555', linestyle=(0, (3, 2)), linewidth=.9, zorder=1)
    ax[1].axhline(4.95, color='#dddddd', linewidth=.6, zorder=0)
    for i, key in enumerate(A_METHODS[1:]):
        add_interval(ax[2], (7., 2.5)[i], {'mean': a[key]['covariance_difference']}, LABELS[key],
                     covariance=True)
    ax[2].set_title('(c) Covariance (log)', pad=7, fontsize=9.8)
    ax[2].set(xlabel='Covariance difference', xscale='log',
              xlim=(1e-14, 1), ylim=(0, 10), yticks=[])
    ax[2].set_xticks([1e-12, 1e-6, 1])
    ax[2].xaxis.set_major_formatter(LogFormatterMathtext())
    ax[2].minorticks_off()
    legend(fig, ['Joint', 'RPI', 'Per-step map'])
    finish(fig, ax, out, NAMES[0])


def plot_b(b, out):
    fig, axes = canvas(3, 7.16, 3.65)
    centers = [15., 11., 7., 3.]
    titles = ('(a) Common error', '(b) Relative error', '(c) Uncertainty')
    labels = ('RMSE (m)', 'RMSE (m)', 'Centroid uncertainty (m)')
    for pi, (ax, metric, title, xlabel) in enumerate(zip(axes, B_METRICS, titles, labels)):
        for i, key in enumerate(B_METHODS):
            for j, sc in enumerate(SCENARIOS):
                add_interval(ax, centers[j]+(1.35-i*.9), b[(sc, key, metric)], B_LABELS[key],
                             digits=4 if pi == 1 else 3)
        ax.set_title(title, pad=7, fontsize=9.8)
        ax.set(xlabel=xlabel, ylim=(.8, 16.8), yticks=[])
        for y in [13., 9., 5.]:
            ax.axhline(y, color='#dddddd', linewidth=.6, zorder=0)
    common_upper = max(1.5, np.ceil((max(b[(sc, key, B_METRICS[0])]['ci95'][1]
                                       for sc in SCENARIOS for key in B_METHODS)+.05)/.5)*.5)
    axes[0].set(xlim=(0, common_upper), xticks=np.arange(0, common_upper+.1, .5),
                yticks=centers, yticklabels=['Correct\nmap', 'Shift\n+1 m', 'Atten.\n+20%', 'Both\nerrors'])
    axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    relative_low = min(.057, min(b[(sc, key, B_METRICS[1])]['ci95'][0]
                                 for sc in SCENARIOS for key in B_METHODS)-.001)
    relative_high = max(.067, max(b[(sc, key, B_METRICS[1])]['ci95'][1]
                                  for sc in SCENARIOS for key in B_METHODS)+.002)
    axes[1].set_xlim(relative_low, relative_high)
    axes[1].set_xticks(np.linspace(relative_low+.001, relative_high-.001, 3))
    axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axes[1].set_xlabel('RMSE (m; zoomed)')
    uncertainty_upper = max(.8, np.ceil(max(b[(sc, key, B_METRICS[2])]['ci95'][1]
                                            for sc in SCENARIOS for key in B_METHODS)*1.15/.2)*.2)
    axes[2].set_xlim(0, uncertainty_upper)
    axes[2].set_xticks(np.linspace(0, uncertainty_upper, 3))
    axes[2].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    legend(fig, ['Scalar', 'RPI (fixed map)', 'Joint', 'RPI (uncertain map)'],
           joint_label='Joint (fixed map)')
    finish(fig, axes, out, NAMES[1])


def plot_c(c, out):
    fig, axes = canvas(2, 7.16, 2.80)
    centers = [7.6, 4.6, 1.6]
    for pi, (ax, axis) in enumerate(zip(axes, ('x', 'y'))):
        for i, key in enumerate(C_METHODS):
            for j, condition in enumerate(MAPS):
                add_interval(ax, centers[j]+(.85-i*.85), c[(condition, key, axis)], C_LABELS[i])
        ax.set_title(f'({"ab"[pi]}) Common {axis}-position', pad=7, fontsize=9.8)
        ax.set(xlabel='RMSE (m)', xlim=(0, .6), ylim=(.1, 9.),
               xticks=[0, .2, .4, .6], yticks=[])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        for y in [6.1, 3.1]: ax.axhline(y, color='#dddddd', linewidth=.6, zorder=0)
    axes[0].set(yticks=centers, yticklabels=['Constant', 'x-varying', '2D varied'])
    legend(fig, C_LABELS)
    finish(fig, axes, out, NAMES[2])


def finish(fig, axes, out, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        for text, y, lo, hi in ax._direct_labels:
            if text.get_window_extent(renderer).x1 > ax.bbox.x1-2:
                text.xy = (lo, y)
                text.set_position((-6, 0))
                text.set_ha('right')
    fig.canvas.draw()
    fig.savefig(out/f'{name}.pdf', facecolor='white')
    fig.savefig(out/f'{name}.png', dpi=300, facecolor='white')
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('figures'))
    parser.add_argument('--data-root', type=Path, default=Path.cwd())
    parser.add_argument('--noise-sweep', type=Path)
    parser.add_argument('--interface-data', type=Path)
    parser.add_argument('--directional-data', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = args.data_root/'results'
    noise_path = args.noise_sweep or results/'gaussian_noise_sweep_20261011/run_7300322/experiment'
    a = load_a(results/'acc2027_revision/experiment_a')
    b = load_b(args.interface_data or results/'field_uncertainty_20261021/experiment')
    c = load_c(args.directional_data or results/'directional_uncertainty_20261022/experiment')
    noise = load_noise(noise_path)
    set_style()
    plot_a(a, args.output, noise)
    plot_b(b, args.output)
    plot_c(c, args.output)


if __name__ == '__main__':
    main()
