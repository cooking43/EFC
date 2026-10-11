"""Scientific figures for the Chinese draft, without altering experiment files."""
from pathlib import Path
import csv
import itertools
import json
import hashlib
import os
import math
from html.parser import HTMLParser

OUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path('F:/JMSE/h1_2_dynamic_deck_starter/experiments/reviewer_experiments/runs')
FINAL = RUNS / 'FINAL_REVIEWER_EVIDENCE_20260930'
METHODS = ['Dynamic-LSTM-47', 'State-Enhanced-LSTM', 'Prediction-to-Policy',
           'Current-State-Residual', 'Predictive-Residual', 'Oracle-Predictive-Residual']
NAMES = ['Basic LSTM', 'Enhanced LSTM', 'Prediction Input Only',
         'Current-Roll Knee Residual', 'EFC', 'Oracle Knee Residual']
COLORS = dict(zip(METHODS, ['#334155', '#0891b2', '#2563eb', '#ea580c', '#15803d', '#9333ea']))
LABELS = dict(zip(METHODS, NAMES))
CORE = METHODS[2:5]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none'})
SOURCES = set()
OUTPUTS = []
ANALYSIS = {}
METRICS = ['success', 'elapsed_s', 'base_tilt_rmse_rad',
           'contacting_foot_slip_rms_m_s', 'action_delta_rms', 'return']
REPLICATES = 20000
RNG = np.random.default_rng(20261010)


def read(path):
    SOURCES.add(path)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save(fig, stem, methods):
    handles, labels = [], []
    for ax in fig.axes:
        ax.grid(alpha=.18)
        for h, name in zip(*ax.get_legend_handles_labels()):
            if name not in labels:
                handles.append(h); labels.append(name)
    if labels:
        fig.legend(handles, labels, loc='lower center', ncol=min(3, len(labels)),
                   bbox_to_anchor=(.5, .005), frameon=False)
    fig.tight_layout(rect=(0, .075, 1, .96))
    for extension in ['png', 'svg']:
        path = OUT / f'{stem}_v3.{extension}'
        fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
        OUTPUTS.append(path)
    plt.close(fig)


def smooth(values, window=31):
    # A trailing mean uses only completed updates, not future observations.
    values = np.asarray(values)
    sums = np.r_[0., np.cumsum(values)]
    return np.array([(sums[i + 1] - sums[max(0, i + 1 - window)]) / min(window, i + 1)
                     for i in range(len(values))])


def interval(values, hierarchical=False):
    values = np.asarray(values, dtype=float)
    assert np.isfinite(values).all()
    if hierarchical:
        # Each selected training cluster receives its own episode resample,
        # including when the same cluster is selected more than once.
        n_cluster, n_episode = values.shape
        clusters = RNG.integers(n_cluster, size=(REPLICATES, n_cluster, 1))
        episodes = RNG.integers(n_episode, size=(REPLICATES, n_cluster, n_episode))
        draws = values[clusters, episodes].mean(axis=(1, 2))
    else:
        draws = values[RNG.integers(values.size, size=(REPLICATES, values.size))].mean(axis=1)
    low, high = np.quantile(draws, [.025, .975])
    return {'mean': float(values.mean()), 'ci95_low': float(low), 'ci95_high': float(high)}


def summarize_episodes(rows, keys, metrics=METRICS, hierarchical=False):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in sorted(groups.items()):
        for metric in metrics:
            if hierarchical:
                values = np.array([[float(r[metric]) for r in sorted(group, key=lambda r: int(r['eval_seed']))
                                    if int(r['train_seed']) == seed] for seed in [37, 73, 109]])
                assert values.shape == (3, 30)
            else:
                values = np.array([float(r[metric]) for r in group])
                assert values.size == 30
            result.append(dict(zip(keys, key), metric=metric, n=len(group),
                               completed=sum(int(r['success']) for r in group),
                               **interval(values, hierarchical)))
    return result


def analyze_current_data():
    p6 = read(RUNS / 'P6_three_method_three_train_seed_ID_60s_30seeds/evaluation_episodes.csv')
    f1 = read(RUNS / 'F1_seed37_six_method_ID_60s_30seeds/evaluation_episodes.csv')
    f2 = read(RUNS / 'F2_seed37_six_method_OOD_7_10deg_60s_30seeds/evaluation_episodes.csv')
    f4 = read(RUNS / 'F4_seed37_sensor_stress_5deg_60s_30seeds/sensor_stress_episodes.csv')
    pred = read(RUNS / 'F3_roll_predictor_formal_30seeds/prediction_episode_metrics.csv')
    assert [len(p6), len(f1), len(f2), len(f4), len(pred)] == [1080, 720, 360, 750, 1620]
    for rows in [p6, f1, f2]:
        identity = [(r['method'], r['train_seed'], r['amplitude_deg'], r['eval_seed']) for r in rows]
        assert len(identity) == len(set(identity))
        assert set(int(r['eval_seed']) for r in rows) == set(range(5001, 5031))
    ANALYSIS['core'] = summarize_episodes(p6, ['method', 'amplitude_deg'], hierarchical=True)
    ANALYSIS['per_training_seed'] = summarize_episodes(p6, ['method', 'train_seed', 'amplitude_deg'])
    ANALYSIS['single_policy'] = summarize_episodes(f1 + f2, ['method', 'amplitude_deg'])
    f4core = [r for r in f4 if r['method'] in CORE]
    ANALYSIS['stress'] = summarize_episodes(f4core, ['method', 'scenario'],
                                          METRICS + ['roll_prediction_rmse_rad'])
    predictions = []
    for method in sorted(set(r['method'] for r in pred)):
        for horizon in [100, 200]:
            values = np.array([np.mean([float(r['rmse_rad']) for r in pred
                                       if r['method'] == method and int(r['horizon_ms']) == horizon
                                       and int(r['seed']) == seed]) for seed in range(5001, 5031)])
            predictions.append({'method': method, 'horizon_ms': horizon, **interval(values)})
    ANALYSIS['prediction'] = predictions
    lookup = {(r['method'], int(r['train_seed']), float(r['amplitude_deg']), int(r['eval_seed'])): r for r in p6}
    paired = []
    for reference in CORE[:2]:
        for amplitude in [0., 1., 3., 5.]:
            for metric in METRICS:
                a = np.array([[float(lookup[('Predictive-Residual', s, amplitude, e)][metric])
                               for e in range(5001, 5031)] for s in [37, 73, 109]])
                b = np.array([[float(lookup[(reference, s, amplitude, e)][metric])
                               for e in range(5001, 5031)] for s in [37, 73, 109]])
                diff = a - b
                seed_diff = diff.mean(axis=1)
                observed = abs(float(seed_diff.mean()))
                null = [abs(float(np.mean(seed_diff * signs)))
                        for signs in itertools.product([-1, 1], repeat=3)]
                sd = float(seed_diff.std(ddof=1))
                record = {'reference': reference, 'amplitude_deg': amplitude, 'metric': metric,
                          'training_seed_differences': seed_diff.tolist(),
                          'effect_training_seed_dz': float(seed_diff.mean() / sd) if sd > 0 else None,
                          'exact_training_seed_p': float(np.mean(np.array(null) >= observed - 1e-15)),
                          **interval(diff, hierarchical=True)}
                if metric == 'success':
                    wins = int(((a == 1) & (b == 0)).sum())
                    losses = int(((a == 0) & (b == 1)).sum())
                    n = wins + losses
                    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.
                    record['descriptive_mcnemar'] = {'efc_only': wins, 'reference_only': losses, 'p': p}
                paired.append(record)
    ANALYSIS['paired'] = paired
    ANALYSIS['row_counts'] = {'P6': len(p6), 'F1': len(f1), 'F2': len(f2), 'F4': len(f4), 'F3_episode': len(pred)}
    ANALYSIS['data_basis'] = 'Author-provided updated records; statistical calculations do not independently attest experimental provenance.'
    ANALYSIS['bootstrap'] = {'replicates': REPLICATES, 'random_seed': 20261010,
                             'core': 'training clusters then matched disturbance episodes',
                             'single_policy': 'disturbance episodes conditional on one frozen policy'}
    (OUT / 'chapter_statistics_20261010.json').write_text(json.dumps(ANALYSIS, indent=2), encoding='utf-8')
    for name in ['core', 'per_training_seed', 'single_policy', 'stress', 'prediction']:
        rows = ANALYSIS[name]
        with (OUT / f'chapter_{name}_20261010.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def training():
    records = read(RUNS / 'P4_combined_seed37_comparison/combined_training_updates.csv')
    metrics = [('mean_rollout_reward', 'Mean total reward per step'),
               ('mean_world_upright_reward', 'World-upright contribution'),
               ('mean_support_relative_height_reward', 'Support-height contribution'),
               ('mean_joint_deviation_penalty', 'Joint-deviation contribution'),
               ('mean_action_delta_penalty', 'Action-change contribution'),
               ('mean_angular_velocity_penalty', 'Angular-velocity contribution'),
               ('mean_slip_penalty', 'Foot-slip contribution')]
    assert all(float(row['mean_fall_penalty']) == 0 for row in records)
    fig = plt.figure(figsize=(13, 12))
    grid = fig.add_gridspec(4, 2, height_ratios=[1.2, 1, 1, 1])
    axes = [fig.add_subplot(grid[0, :])]
    axes.extend(fig.add_subplot(grid[row, col]) for row in range(1, 4) for col in range(2))
    for panel, (ax, (key, label)) in enumerate(zip(axes, metrics)):
        for method in METHODS:
            group = sorted([r for r in records if r['method'] == method], key=lambda r: int(r['global_step']))
            assert len(group) == 1024
            ax.plot([int(r['global_step']) / 1e6 for r in group], smooth([float(r[key]) for r in group]),
                    label=LABELS[method], color=COLORS[method], linewidth=1.9)
        ax.set(xlabel='Environment steps (millions)', ylabel='Mean reward per step',
               title=f'({chr(97 + panel)}) {label}')
    fig.suptitle('Training reward and components (one run per method)', fontsize=14)
    save(fig, 'fig4_2_training_reward_components', METHODS)


def predictor():
    methods = ['Persistence', 'First-Order', 'Second-Order', 'Filtered-Second-Order', 'Causal-History', 'Oracle']
    labels = ['Persistence', 'First-order', 'Raw second-order', 'Filtered second-order', 'Causal history', 'Oracle']
    colors = ['#64748b', '#2563eb', '#15803d', '#ea580c', '#9333ea', '#111827']
    stats = ANALYSIS['prediction']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    time_values, traces = [], {name: [] for name in ['truth', 'Second-Order', 'Filtered-Second-Order']}
    frame_path = RUNS / 'F3_roll_predictor_formal_30seeds/prediction_frames.csv'
    SOURCES.add(frame_path)
    with frame_path.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if int(row['seed']) > 5001:
                break
            if float(row['amplitude_deg']) == 3 and int(row['horizon_ms']) == 100 and 2 <= float(row['time_s']) <= 7:
                if row['method'] == 'Second-Order':
                    time_values.append(float(row['time_s']))
                    traces['truth'].append(float(row['truth_roll_rad']))
                if row['method'] in traces:
                    traces[row['method']].append(float(row['prediction_roll_rad']))
    # Amplitude-major generation puts seed5001 amplitude3 after other seeds at amplitude1;
    # if the early read missed it, read exactly until the requested stratum finishes.
    if not time_values:
        with frame_path.open(encoding='utf-8', newline='') as stream:
            found = False
            for row in csv.DictReader(stream):
                selected = int(row['seed']) == 5001 and float(row['amplitude_deg']) == 3 and int(row['horizon_ms']) == 100
                if found and not selected:
                    break
                if selected:
                    found = True
                    if 2 <= float(row['time_s']) <= 7:
                        if row['method'] == 'Second-Order':
                            time_values.append(float(row['time_s']))
                            traces['truth'].append(float(row['truth_roll_rad']))
                        if row['method'] in traces:
                            traces[row['method']].append(float(row['prediction_roll_rad']))
    for key, label, color, style in [('truth', 'Future-roll target', '#111827', '-'),
                                     ('Second-Order', 'Raw second-order', '#15803d', '--'),
                                     ('Filtered-Second-Order', 'Filtered second-order', '#ea580c', '-')]:
        axes[0].plot(time_values, traces[key], label=label, color=color, linestyle=style, linewidth=1.8)
    axes[0].set(title='100 ms forecast (3 deg example)', xlabel='Forecast issue time (s)', ylabel='Roll angle (rad)')
    for method, label, color in zip(methods[:-1], labels[:-1], colors[:-1]):
        group = [r for r in stats if r['method'] == method]
        x = [r['horizon_ms'] for r in group]
        axes[1].plot(x, [r['mean'] for r in group], marker='o', label=label, color=color)
        axes[1].fill_between(x, [r['ci95_low'] for r in group], [r['ci95_high'] for r in group], color=color, alpha=.13)
    axes[1].set(title='Matched-seed macro RMSE | 95% bootstrap CI', xlabel='Prediction horizon (ms)', ylabel='Roll prediction RMSE (rad)', yscale='log', xticks=[100, 200])
    axes[1].text(.67, .04, 'Oracle: zero error\n(not on log axis)', transform=axes[1].transAxes, fontsize=9)
    fig.suptitle('Short-horizon platform-roll prediction', fontsize=14)
    save(fig, 'fig4_3_prediction_accuracy', [])


def core():
    data = ANALYSIS['core']
    metrics = [('success', 'Completion rate'), ('elapsed_s', 'RMST (s)'),
               ('base_tilt_rmse_rad', 'Base tilt RMSE (rad)'),
               ('contacting_foot_slip_rms_m_s', 'Foot-slip RMS (m/s)'),
               ('action_delta_rms', 'Action-change RMS'), ('return', 'Episode return')]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for ax, (metric, label) in zip(axes.flat, metrics):
        for method in CORE:
            rows = sorted([r for r in data if r['method'] == method and r['metric'] == metric], key=lambda r: float(r['amplitude_deg']))
            x = [float(r['amplitude_deg']) for r in rows]
            ax.plot(x, [float(r['mean']) for r in rows], marker='o', color=COLORS[method], label=LABELS[method])
            ax.fill_between(x, [float(r['ci95_low']) for r in rows], [float(r['ci95_high']) for r in rows], color=COLORS[method], alpha=.13)
        ax.set(xlabel='Combined roll/pitch amplitude (deg)', ylabel=label, xticks=[0, 1, 3, 5])
    fig.suptitle('Core comparison (3 independent training runs, 30 matched disturbances, 95% CI)', fontsize=13)
    save(fig, 'fig4_4_core_multiseed_comparison', CORE)


def id_ood():
    data = ANALYSIS['single_policy']
    metrics = [('success', 'Completion rate'), ('elapsed_s', 'RMST (s)'),
               ('base_tilt_rmse_rad', 'Base tilt RMSE (rad)'),
               ('contacting_foot_slip_rms_m_s', 'Foot-slip RMS (m/s)'),
               ('action_delta_rms', 'Action-change RMS')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (metric, label) in zip(axes.flat, metrics):
        for method in METHODS:
            rows = sorted([r for r in data if r['method'] == method and r['metric'] == metric], key=lambda r: float(r['amplitude_deg']))
            x = [float(r['amplitude_deg']) for r in rows]
            ax.plot(x, [float(r['mean']) for r in rows], marker='o', color=COLORS[method], label=LABELS[method])
            ax.fill_between(x, [float(r['ci95_low']) for r in rows], [float(r['ci95_high']) for r in rows], color=COLORS[method], alpha=.12)
        ax.axvline(5, color='#64748b', linestyle='--', linewidth=1)
        ax.set(xlabel='Combined roll/pitch amplitude (deg)', ylabel=label, xticks=[0, 1, 3, 5, 7, 10])
    axes.flat[-1].axis('off')
    fig.suptitle('Six-method comparison (one frozen policy per method, 30 matched 60 s episodes, 95% CI)', fontsize=13)
    save(fig, 'fig4_7_id_ood_six_method', METHODS)


def stress():
    data = ANALYSIS['stress']
    scenarios = ['clean', 'noise_nominal', 'noise_high', 'bias_nominal', 'bias_high', 'latency_20ms', 'latency_40ms', 'latency_80ms']
    ticks = ['Clean', 'Noise\n0.1 deg', 'Noise\n0.5 deg', 'Bias\n0.25 deg', 'Bias\n1 deg', 'Delay\n20 ms', 'Delay\n40 ms', 'Delay\n80 ms']
    metrics = [('success', 'Completion rate'), ('elapsed_s', 'RMST (s)'),
               ('base_tilt_rmse_rad', 'Base tilt RMSE (rad)'),
               ('contacting_foot_slip_rms_m_s', 'Foot-slip RMS (m/s)'),
               ('action_delta_rms', 'Action-change RMS'), ('roll_prediction_rmse_rad', '100 ms forecast RMSE (rad)')]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (metric, label) in zip(axes.flat, metrics):
        for method in (['Predictive-Residual'] if metric == 'roll_prediction_rmse_rad' else CORE):
            rows = [next(r for r in data if r['method'] == method and r['metric'] == metric and r['scenario'] == scenario) for scenario in scenarios]
            x = range(len(rows))
            ax.plot(x, [float(r['mean']) for r in rows], marker='o', color=COLORS[method], label=LABELS[method])
            ax.fill_between(x, [float(r['ci95_low']) for r in rows], [float(r['ci95_high']) for r in rows], color=COLORS[method], alpha=.13)
        ax.set(ylabel=label, xticks=list(range(8)), xticklabels=ticks)
        ax.tick_params(axis='x', labelsize=9)
        if metric == 'success':
            ax.set_ylim(0, 1.05)
        if metric == 'elapsed_s':
            ax.set_ylim(0, 62)
    fig.suptitle('Roll-observation sensitivity (one frozen policy per method, 5 deg, 30 matched episodes, 95% CI)', fontsize=13)
    save(fig, 'fig4_8_sensor_sensitivity', CORE)


def seed_detail():
    data = ANALYSIS['per_training_seed']
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for offset, method in zip([-.22, 0, .22], CORE):
        rows = [next(r for r in data if r['method'] == method and int(r['train_seed']) == seed
                     and float(r['amplitude_deg']) == 5 and r['metric'] == 'elapsed_s') for seed in [37, 73, 109]]
        x = np.arange(3) + offset
        means = np.array([r['mean'] for r in rows])
        ax.bar(x, means - 40, bottom=40, width=.21, color=COLORS[method], label=LABELS[method], alpha=.85)
        ax.errorbar(x, means, yerr=[means - [r['ci95_low'] for r in rows],
                                  [r['ci95_high'] for r in rows] - means], fmt='none', color=COLORS[method], capsize=4)
    ax.set(xticks=np.arange(3), xticklabels=['Run 1', 'Run 2', 'Run 3'], ylabel='RMST (s)',
           ylim=(40, 64), yticks=[40, 45, 50, 55, 60])
    fig.suptitle('5 deg: RMST by independent training run (episode-bootstrap 95% CI)', fontsize=13)
    save(fig, 'fig4_5_training_run_detail', CORE)


def paired_detail():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric, label in zip(axes, METRICS[2:5], ['Tilt RMSE difference (rad)', 'Slip difference (m/s)', 'Action-change difference']):
        for y, reference in enumerate(CORE[:2]):
            row = next(r for r in ANALYSIS['paired'] if r['reference'] == reference and r['amplitude_deg'] == 5 and r['metric'] == metric)
            mean = row['mean']
            ax.errorbar(mean, y, xerr=[[mean-row['ci95_low']], [row['ci95_high']-mean]],
                        fmt='o', color=COLORS[reference], capsize=5, linewidth=2)
        ax.axvline(0, color='#64748b', linestyle='--')
        ax.set(xlabel=label, yticks=[0, 1], yticklabels=['vs Prediction Input', 'vs Current-Roll'], ylim=(-.6, 1.6))
    fig.suptitle('EFC minus comparator at 5 deg | hierarchical 95% CI; negative = lower error', fontsize=13)
    save(fig, 'fig4_6_paired_effects', [])


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_draft():
    class Tables(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tables, self.images, self.current, self.row, self.cell = [], [], None, [], None
        def handle_starttag(self, tag, attrs):
            if tag == 'table': self.current = []
            if tag == 'tr': self.row = []
            if tag in ['td', 'th']: self.cell = ''
            if tag == 'img': self.images.append(dict(attrs)['src'])
        def handle_data(self, text):
            if self.cell is not None: self.cell += text
        def handle_endtag(self, tag):
            if tag in ['td', 'th']:
                self.row.append(self.cell.strip()); self.cell = None
            if tag == 'tr' and self.current is not None: self.current.append(self.row)
            if tag == 'table': self.tables.append(self.current); self.current = None
    parser = Tables()
    html_file = OUT / '第四章实验中文讨论稿.html'
    parser.feed(html_file.read_text(encoding='utf-8'))
    assert len(parser.tables) == 9 and len(parser.images) == 8
    assert all((OUT / name).is_file() for name in parser.images)
    assert all('_v2' not in name for name in parser.images)
    def get(section, method, metric, amplitude=5., scenario=None):
        return next(r for r in ANALYSIS[section] if r['method'] == method and r['metric'] == metric
                    and (r.get('scenario') == scenario if scenario is not None else float(r['amplitude_deg']) == amplitude))
    def ci(row, precision, scale=1.):
        return f"{row['mean']*scale:.{precision}f} [{row['ci95_low']*scale:.{precision}f}, {row['ci95_high']*scale:.{precision}f}]"
    predictors = ['Persistence','First-Order','Second-Order','Filtered-Second-Order','Causal-History','Oracle']
    for cells, method in zip(parser.tables[2][1:], predictors):
        for cell, horizon in zip(cells[1:], [100, 200]):
            row = next(r for r in ANALYSIS['prediction'] if r['method'] == method and r['horizon_ms'] == horizon)
            assert cell == ('0' if method == 'Oracle' else ci(row, 3, 1000)), (method, cell)
    for cells, method in zip(parser.tables[3][1:], CORE):
        for cell, amplitude in zip(cells[1:5], [0, 1, 3, 5]):
            assert cell == f"{get('core', method, 'success', amplitude)['completed']}/90"
        success = get('core', method, 'success')
        expected_success = f"{success['mean']*100:.1f}% [{success['ci95_low']*100:.1f}%, {success['ci95_high']*100:.1f}%]"
        assert cells[5] == expected_success
        assert cells[6] == ci(get('core', method, 'elapsed_s'), 2)
    for cells, method in zip(parser.tables[4][1:], CORE):
        for cell, metric, precision in zip(cells[1:], METRICS[2:5], [5, 4, 5]):
            assert cell == ci(get('core', method, metric), precision), (method, metric, cell)
    for cells, (reference, metric) in zip(parser.tables[5][1:], itertools.product(CORE[:2], METRICS[2:5])):
        row = next(r for r in ANALYSIS['paired'] if r['reference'] == reference and r['metric'] == metric and r['amplitude_deg'] == 5)
        precision = 5 if metric == METRICS[3] else 6
        assert cells[2].replace('−', '-') == ci(row, precision)
        assert cells[3].replace('−', '-') == f"{row['effect_training_seed_dz']:.3f}"
        assert cells[4] == f"{row['exact_training_seed_p']:.2f}"
    for cells, method in zip(parser.tables[6][1:], METHODS):
        assert cells[1] == f"{get('single_policy', method, 'success')['completed']}/30"
        for cell, metric, precision in zip(cells[2:], METRICS[1:5], [2, 5, 4, 5]):
            assert cell == f"{get('single_policy', method, metric)['mean']:.{precision}f}"
    for cells, method in zip(parser.tables[7][1:], METHODS):
        for count_index, time_index, amplitude in [(1, 2, 7), (3, 4, 10)]:
            assert cells[count_index] == f"{get('single_policy', method, 'success', amplitude)['completed']}/30"
            assert cells[time_index] == f"{get('single_policy', method, 'elapsed_s', amplitude)['mean']:.2f}"
    scenarios = ['clean','noise_nominal','noise_high','bias_nominal','bias_high','latency_20ms','latency_40ms','latency_80ms']
    clean = get('stress', 'Predictive-Residual', 'roll_prediction_rmse_rad', scenario='clean')['mean']
    for cells, scenario in zip(parser.tables[8][1:], scenarios):
        value = get('stress', 'Predictive-Residual', 'roll_prediction_rmse_rad', scenario=scenario)['mean']
        assert cells[1] == f'{value:.5f}' and cells[2] == f'{value/clean:.2f} 倍'
        for cell, method in zip(cells[3:], CORE):
            assert cell == f"{get('stress', method, 'success', scenario=scenario)['completed']}/30"
    manifest = json.loads((OUT / 'plot_manifest.json').read_text(encoding='utf-8'))
    assert all(digest(Path(path)) == expected for path, expected in manifest['sources_sha256'].items())
    assert all(digest(OUT / path) == expected for path, expected in manifest['outputs_sha256'].items())
    manuscript = Path('F:/JMSE/JMSE/JMSE.tex').read_text(encoding='utf-8')
    backup = Path('F:/JMSE/JMSE/response/backups/JMSE_before_experiment_chapter_20261010.tex')
    latex_scope_ok = False
    if backup.is_file():
        old = backup.read_text(encoding='utf-8')
        start, end = r'\section{Experiments and Results Analysis}', r'\section{Conclusion}'
        prefix, rest = manuscript.split(start, 1)
        _, suffix = rest.split(end, 1)
        old_prefix, old_rest = old.split(start, 1)
        _, old_suffix = old_rest.split(end, 1)
        latex_scope_ok = prefix == old_prefix and suffix == old_suffix
    else:
        latex_scope_ok = digest(Path('F:/JMSE/JMSE/JMSE.tex')) == '43de74f2fcf28c4cce3b5757a746e64a96d79f029622e1436221f636c757ec61'
    assert latex_scope_ok
    validation = {'figures': 8, 'tables': 9, 'numeric_result_tables_matched_recomputed_statistics': True,
                  'source_hashes_match_manifest': True, 'output_hashes_match_manifest': True,
                  'latex_only_experiment_chapter_changed': True, 'browser_layout_check': 'not performed: local file protocol blocked',
                  'scope': 'computational consistency only, not independent verification of experimental provenance'}
    (OUT / 'draft_validation_20261010.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    print('Draft checks passed: 8 figures, 9 tables, numeric tables and file hashes', flush=True)


if __name__ == '__main__':
    if '--training-rmst-only' in __import__('sys').argv:
        ANALYSIS.update(json.loads((OUT / 'chapter_statistics_20261010.json').read_text(encoding='utf-8')))
        seed_detail()
        print('Training-repeat RMST figure regenerated; existing statistics unchanged', flush=True)
        raise SystemExit(0)
    if '--check' in __import__('sys').argv:
        ANALYSIS.update(json.loads((OUT / 'chapter_statistics_20261010.json').read_text(encoding='utf-8')))
        validate_draft()
        raise SystemExit(0)
    analyze_current_data(); print('Current episode statistics complete', flush=True)
    training(); print('Training figure complete', flush=True)
    predictor(); print('Prediction figure complete', flush=True)
    core(); id_ood(); stress(); seed_detail(); paired_detail()
    manifest = {'method_colors': COLORS, 'method_labels': LABELS,
                'sources_sha256': {str(p): digest(p) for p in sorted(SOURCES)},
                'outputs_sha256': {p.name: digest(p) for p in OUTPUTS},
                'training_smoothing': 'trailing 31 updates; seed37 only, no CI',
                'revision': '2026-10-10: recomputed summaries, paired analyses and figures from author-provided updated episode CSVs; no source CSV changes',
                'data_basis': ANALYSIS['data_basis'],
                'training_constant_component': 'all 6144 recorded mean_fall_penalty values are zero; omitted from main figure, retained in original records and v1 figure',
                'training_axis': 'cumulative environment control steps, 1024 per update, final 1048576; not wall-clock time',
                'training_seed_ids': [37, 73, 109],
                'evaluation_seed_ids': list(range(5001, 5031)),
                'core_ci': '20000 two-level bootstrap replicates; training clusters then disturbance episodes; paired differences retain matching',
                'prediction_ci': '20000 bootstrap replicates, seed20261010, resample 30 evaluation seeds; fixed amplitude mean',
                'id_ood_and_stress_ci': 'recomputed conditional episode-bootstrap intervals, 20000 replicates',
                'excluded_prediction_horizon': 'nominal50ms has actual40ms target labels; excluded from main prediction figure'}
    for path in [OUT / '第四章实验中文讨论稿.html', OUT / 'chapter_statistics_20261010.json',
                 OUT / 'fig4_1_efc_dynamic_sequence.png', *sorted(OUT.glob('chapter_*_20261010.csv'))]:
        manifest['outputs_sha256'][path.name] = digest(path)
    (OUT / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    validate_draft()
    print('All draft figures complete', flush=True)
