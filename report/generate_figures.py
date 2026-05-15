import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    'font.family':       'DejaVu Serif',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        120,
    'savefig.dpi':       180,
    'savefig.bbox':      'tight',
    'legend.frameon':    False,
})

HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(os.path.dirname(HERE), 'results')
FIGS_DIR  = os.path.join(HERE, 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

raw     = pd.read_csv(os.path.join(DATA_DIR, 'raw_results.csv'))
summ    = pd.read_csv(os.path.join(DATA_DIR, 'summary_results.csv'))
pairs   = pd.read_csv(os.path.join(DATA_DIR, 'pairwise_tests.csv'))

C_META   = '#0b5cad'
C_NOMETA = '#c45a30'
C_GREY   = '#444444'
C_ACCENT = '#1f8a4d'

SPEEDS = ['slow', 'medium', 'fast', 'very_fast']
SPEED_LABEL = {'slow': 'slow\n(μ=20)', 'medium': 'medium\n(μ=10)',
               'fast': 'fast\n(μ=5)',  'very_fast': 'very fast\n(μ=2)'}

def fig_arch():
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')

    def box(x, y, w, h, text, fc='#eef3fb', ec=C_META, fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle='round,pad=0.04,rounding_size=0.10',
            fc=fc, ec=ec, lw=1.4))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center',
                fontsize=fs, color='black')

    def arrow(x1, y1, x2, y2, label=None, lblpos=None, color=C_GREY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
            arrowstyle='-|>', mutation_scale=12, lw=1.2, color=color))
        if label and lblpos:
            ax.text(lblpos[0], lblpos[1], label, fontsize=8.5,
                    color=color, ha='center')

    box(0.2, 4.4, 2.6, 1.1, 'BSE market\n(supply / demand schedule)', fc='#fff5e9', ec=C_NOMETA)
    box(0.2, 2.6, 2.6, 1.1, 'Regime switcher\n(Gaussian dwell-time)', fc='#fff5e9', ec=C_NOMETA)
    box(0.2, 0.8, 2.6, 1.1, 'Background traders\nZIC / ZIP / SHVR',  fc='#fff5e9', ec=C_NOMETA)

    box(3.4, 4.4, 3.0, 1.1, 'Feature extractor\n(momentum, vol, range, ρ)', fc='#eef3fb', ec=C_META)
    box(3.4, 2.6, 3.0, 1.1, 'Gaussian HMM\n(Forward algorithm)', fc='#eef3fb', ec=C_META)
    box(3.4, 0.8, 3.0, 1.1, 'Meta-learner\n(rolling error-rate monitor)', fc='#eef3fb', ec=C_META)

    box(7.0, 4.4, 2.8, 1.1, 'Risk manager\n(veto under low conf / DD)', fc='#eaf6ee', ec=C_ACCENT)
    box(7.0, 2.6, 2.8, 1.1, 'Agent selector\n(regime → strategy)',     fc='#eaf6ee', ec=C_ACCENT)
    box(7.0, 0.8, 2.8, 1.1, 'Momentum / Contrarian /\nMarket-maker agent', fc='#eaf6ee', ec=C_ACCENT)

    arrow(2.8, 4.95, 3.4, 4.95, 'tape', (3.1, 5.15))
    arrow(2.8, 3.15, 3.4, 3.15, 'features', (3.1, 3.35))
    arrow(6.4, 3.15, 7.0, 3.15, 'regime', (6.7, 3.35))
    arrow(6.4, 1.35, 7.0, 1.35, 'retrain', (6.7, 1.55))
    arrow(8.4, 4.4, 8.4, 3.7, 'veto / OK', (8.95, 4.05))
    arrow(8.4, 2.6, 8.4, 1.9)

    arrow(4.9, 2.6, 4.9, 1.9, 'error', (5.25, 2.25))
    arrow(4.9, 0.8, 4.9, 0.45, color=C_META)
    ax.text(4.9, 0.2, 'rolling window retrain (Baum–Welch)',
            ha='center', fontsize=8.5, color=C_META)

    plt.savefig(os.path.join(FIGS_DIR, 'fig_arch.png'))
    plt.close()

def fig_acc_vs_speed():
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    xs = np.arange(len(SPEEDS))
    for meta, col, lbl in [(True, C_META, 'meta-learning on'),
                           (False, C_NOMETA, 'meta-learning off')]:
        ys, errs = [], []
        for sp in SPEEDS:
            row = summ[(summ['speed']==sp) & (summ['meta']==meta) &
                       (summ['experiment']=='speed_sweep')].iloc[0]
            ys.append(row['hmm_accuracy_mean'])
            errs.append(row['hmm_accuracy_ci95'])
        ax.errorbar(xs, ys, yerr=errs, marker='o', ms=6, capsize=4,
                    lw=1.6, color=col, label=lbl)
    ax.set_xticks(xs); ax.set_xticklabels([SPEED_LABEL[s] for s in SPEEDS])
    ax.set_ylabel('HMM accuracy')
    ax.set_title('Classification accuracy across regime-switching speeds')
    ax.set_ylim(0.6, 0.9); ax.grid(alpha=0.25)
    ax.legend(loc='lower left')
    plt.savefig(os.path.join(FIGS_DIR, 'fig_acc_vs_speed.png'))
    plt.close()

def fig_acc_vs_sesslen():
    sl_rows = summ[summ['experiment']=='session_sweep'].sort_values('session_length')
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    xs = sl_rows['session_length'].values
    ys = sl_rows['hmm_accuracy_mean'].values
    errs = sl_rows['hmm_accuracy_ci95'].values
    ax.errorbar(xs, ys, yerr=errs, marker='s', ms=7, capsize=4,
                lw=1.8, color=C_ACCENT)
    for x, y in zip(xs, ys):
        ax.annotate(f'{y:.1%}', (x, y), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=9, color=C_GREY)
    ax.axhline(1/3, ls=':', color='grey', lw=1)
    ax.text(280, 1/3+0.012, 'chance (1/3)', fontsize=8, color='grey')
    ax.set_xlabel('Session length (simulated seconds)')
    ax.set_ylabel('HMM accuracy')
    ax.set_title('Effect of observation-window length on regime accuracy')
    ax.set_ylim(0.30, 0.86); ax.grid(alpha=0.25)
    plt.savefig(os.path.join(FIGS_DIR, 'fig_acc_vs_sesslen.png'))
    plt.close()

def fig_per_regime_acc():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    regimes = ['trending', 'mean_reverting', 'volatile']
    width = 0.18
    xs = np.arange(len(regimes))
    for i, sp in enumerate(SPEEDS):
        row = summ[(summ['speed']==sp) & (summ['meta']==True) &
                   (summ['experiment']=='speed_sweep')].iloc[0]
        vals = [row[f'acc_{r}_mean'] for r in regimes]
        errs = [row[f'acc_{r}_ci95']  for r in regimes]
        ax.bar(xs + (i-1.5)*width, vals, width,
               yerr=errs, capsize=2,
               color=plt.cm.Blues(0.4 + 0.18*i),
               edgecolor='white', linewidth=0.6,
               label=sp.replace('_', ' '))
    ax.set_xticks(xs); ax.set_xticklabels(['trending', 'mean-reverting', 'volatile'])
    ax.set_ylabel('Per-regime accuracy')
    ax.set_title('Accuracy by regime and switching speed (meta-learning on)')
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.25, axis='y')
    ax.legend(title='speed', loc='lower left', ncol=2, fontsize=8.5)
    plt.savefig(os.path.join(FIGS_DIR, 'fig_per_regime_acc.png'))
    plt.close()

def fig_lag_vs_speed():
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    xs = np.arange(len(SPEEDS))
    for meta, col, lbl in [(True, C_META, 'meta on'),
                           (False, C_NOMETA, 'meta off')]:
        ys, errs = [], []
        for sp in SPEEDS:
            row = summ[(summ['speed']==sp) & (summ['meta']==meta) &
                       (summ['experiment']=='speed_sweep')].iloc[0]
            ys.append(row['detection_lag_mean'])
            errs.append(row['detection_lag_ci95'])
        ax.errorbar(xs, ys, yerr=errs, marker='D', ms=6, capsize=4,
                    lw=1.6, color=col, label=lbl)
    ax.set_xticks(xs); ax.set_xticklabels([SPEED_LABEL[s] for s in SPEEDS])
    ax.set_ylabel('mean detection lag (sessions)')
    ax.set_title('Detection lag at regime transitions')
    ax.grid(alpha=0.25); ax.legend()
    plt.savefig(os.path.join(FIGS_DIR, 'fig_lag_vs_speed.png'))
    plt.close()

def fig_pnl_vs_speed():
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    xs = np.arange(len(SPEEDS)); w = 0.36
    metas   = [summ[(summ['speed']==s)&(summ['meta']==True)&
                    (summ['experiment']=='speed_sweep')].iloc[0]
               for s in SPEEDS]
    nometas = [summ[(summ['speed']==s)&(summ['meta']==False)&
                    (summ['experiment']=='speed_sweep')].iloc[0]
               for s in SPEEDS]
    ax.bar(xs-w/2, [r['total_pnl_mean'] for r in metas], w,
           yerr=[r['total_pnl_ci95'] for r in metas], capsize=3,
           color=C_META, label='meta on', edgecolor='white')
    ax.bar(xs+w/2, [r['total_pnl_mean'] for r in nometas], w,
           yerr=[r['total_pnl_ci95'] for r in nometas], capsize=3,
           color=C_NOMETA, label='meta off', edgecolor='white')
    ax.set_xticks(xs); ax.set_xticklabels([SPEED_LABEL[s] for s in SPEEDS])
    ax.set_ylabel('mean total PnL per run')
    ax.set_title('Trading performance across switching speeds')
    ax.set_ylim(1700, 2400); ax.grid(alpha=0.25, axis='y')
    ax.legend(loc='lower left')
    plt.savefig(os.path.join(FIGS_DIR, 'fig_pnl_vs_speed.png'))
    plt.close()

# -- 7. confidence at transitions vs stable ----------------------------------
def fig_confidence():
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    xs = np.arange(len(SPEEDS)); w = 0.36
    trans  = [summ[(summ['speed']==s)&(summ['meta']==True)&
                   (summ['experiment']=='speed_sweep')].iloc[0]['conf_at_transition_mean']
              for s in SPEEDS]
    stable = [summ[(summ['speed']==s)&(summ['meta']==True)&
                   (summ['experiment']=='speed_sweep')].iloc[0]['conf_at_stable_mean']
              for s in SPEEDS]
    ax.bar(xs-w/2, trans,  w, color='#a23a3a', label='at transition', edgecolor='white')
    ax.bar(xs+w/2, stable, w, color='#3a6ba2', label='stable periods', edgecolor='white')
    ax.set_ylim(0.92, 1.0); ax.set_ylabel('mean HMM posterior')
    ax.set_xticks(xs); ax.set_xticklabels([SPEED_LABEL[s] for s in SPEEDS])
    ax.set_title('HMM confidence: transition sessions vs. stable sessions')
    ax.legend(loc='lower right'); ax.grid(alpha=0.25, axis='y')
    plt.savefig(os.path.join(FIGS_DIR, 'fig_confidence.png'))
    plt.close()

# -- 8. retrain frequency ----------------------------------------------------
def fig_retrains():
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    xs = np.arange(len(SPEEDS))
    vals = [summ[(summ['speed']==s)&(summ['meta']==True)&
                 (summ['experiment']=='speed_sweep')].iloc[0]['n_retrains_mean']
            for s in SPEEDS]
    errs = [summ[(summ['speed']==s)&(summ['meta']==True)&
                 (summ['experiment']=='speed_sweep')].iloc[0]['n_retrains_ci95']
            for s in SPEEDS]
    ax.bar(xs, vals, 0.55, yerr=errs, capsize=4,
           color=C_META, edgecolor='white')
    ax.set_xticks(xs); ax.set_xticklabels([SPEED_LABEL[s] for s in SPEEDS])
    ax.set_ylabel('mean retrains per 100-session run')
    ax.set_title('Meta-learner trigger frequency')
    ax.grid(alpha=0.25, axis='y')
    for x, v in zip(xs, vals):
        ax.text(x, v+0.05, f'{v:.2f}', ha='center', fontsize=9)
    plt.savefig(os.path.join(FIGS_DIR, 'fig_retrains.png'))
    plt.close()

# -- 9. confusion matrix (live phase, all speed-sweep runs) ------------------
# Reconstructed indirectly from per-regime accuracy:
# correct fraction is on the diagonal, off-diagonals are estimated as equal
# split of the remaining mass (this is an approximation since the raw CSV
# does not log the predicted regime per session — we note this in the caption).
def fig_confusion_matrix():
    regimes = ['trending', 'mean_reverting', 'volatile']
    rows = summ[(summ['experiment']=='speed_sweep') & (summ['meta']==True)]
    accs = {r: rows[f'acc_{r}_mean'].mean() for r in regimes}
    cm = np.zeros((3, 3))
    for i, r in enumerate(regimes):
        cm[i, i] = accs[r]
        leftover = 1 - accs[r]
        for j, _ in enumerate(regimes):
            if j != i:
                cm[i, j] = leftover / 2

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
    short = ['trend', 'mean-rev', 'volat.']
    ax.set_xticklabels(short); ax.set_yticklabels(short)
    ax.set_xlabel('predicted'); ax.set_ylabel('true')
    for i in range(3):
        for j in range(3):
            txt = f'{cm[i,j]:.2f}'
            color = 'white' if cm[i,j] > 0.5 else 'black'
            ax.text(j, i, txt, ha='center', va='center', color=color, fontsize=10)
    ax.set_title('Estimated confusion matrix (speed-sweep, meta on)')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(FIGS_DIR, 'fig_confusion_matrix.png'))
    plt.close()


# -- 10. threshold sweep (RQ1 robustness) -----------------------------------
def fig_threshold_sweep():
    path = os.path.join(DATA_DIR, 'threshold_sweep_summary.csv')
    if not os.path.exists(path):
        print('threshold_sweep_summary.csv missing — skipping fig_threshold_sweep')
        return
    ts = pd.read_csv(path).sort_values('threshold')
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    ax = axes[0]
    ax.errorbar(ts['threshold'], ts['hmm_accuracy_mean'],
                yerr=ts['hmm_accuracy_ci95'], marker='o', ms=6,
                capsize=4, lw=1.6, color=C_META, label='accuracy')
    ax.set_xlabel('retrain error-rate threshold')
    ax.set_ylabel('HMM accuracy', color=C_META)
    ax.set_ylim(0.50, 0.85); ax.grid(alpha=0.25)
    ax.tick_params(axis='y', labelcolor=C_META)
    ax2 = ax.twinx()
    ax2.bar(ts['threshold'], ts['n_retrains_mean'], width=0.04,
            alpha=0.25, color=C_NOMETA, label='retrains/run')
    ax2.set_ylabel('mean retrains per 100-session run', color=C_NOMETA)
    ax2.tick_params(axis='y', labelcolor=C_NOMETA)
    ax2.set_ylim(0, max(ts['n_retrains_mean'])*1.4)
    ax2.spines['top'].set_visible(False)
    ax.set_title('Accuracy vs.\\ retrain threshold')

    ax = axes[1]
    ax.errorbar(ts['threshold'], ts['detection_lag_mean'],
                yerr=ts['detection_lag_ci95'], marker='D', ms=6,
                capsize=4, lw=1.6, color=C_GREY)
    ax.set_xlabel('retrain error-rate threshold')
    ax.set_ylabel('mean detection lag (sessions)')
    ax.set_ylim(1.3, 2.0); ax.grid(alpha=0.25)
    ax.set_title('Detection lag vs.\\ retrain threshold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'fig_threshold_sweep.png'))
    plt.close()


if __name__ == '__main__':
    fig_arch()
    fig_acc_vs_speed()
    fig_acc_vs_sesslen()
    fig_per_regime_acc()
    fig_lag_vs_speed()
    fig_pnl_vs_speed()
    fig_confidence()
    fig_retrains()
    fig_confusion_matrix()
    fig_threshold_sweep()
    print('All figures written to', FIGS_DIR)
