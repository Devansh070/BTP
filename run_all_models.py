import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings('ignore')
OUT = 'model_results'
os.makedirs(OUT, exist_ok=True)

measured = {0.005: 43.247, 0.05: 45.468, 0.25: 76.185, 0.7: 154.905, 1.3: 178.603}
own_rows = [('This work (corrected)', e, 180.0 - a, 'simulated', 'own') for e, a in measured.items()]

lit_rows = [
    ('Cone-droplet study (SPC/E)', 0.1881, 138.1, 'simulated graphene'),
    ('Scocchi et al.', 0.200, 127.0, 'target graphene'),
    ('Taherian et al.', 0.205, 127.0, 'target graphene'),
    ('Confinement study (TIP4P/2005)', 0.2703, 96.5, 'simulated graphite wall'),
    ('Bejagam et al.', 0.32505, 97.5, 'target graphene'),
    ('Jaffe et al.', 0.357, 100.7, 'simulated graphene'),
    ('Carlson et al. (inf. cutoff)', 0.3807, 80.0, 'simulated graphene'),
    ('Tummala et al.', 0.3892, 104.0, 'simulated graphene'),
    ('Werder et al. 2003', 0.392, 95.0, 'simulated graphene'),
    ('Liao et al.', 0.3998, 91.83, 'simulated graphene'),
    ('GROMOS53a6 (Carlson et al.)', 0.4247, 82.74, 'simulated graphene'),
    ('Won et al.', 0.4339, 69.1, 'simulated graphite'),
    ('Carlson et al. (0.9 nm)', 0.4391, 80.0, 'target graphene'),
    ('Hummer et al.', 0.477, 57.0, 'simulated graphene'),
    ('Carlson et al. graphite (0.9 nm)', 0.5164, 60.0, 'target graphite'),
    ('Werder et al. (graphite)', 0.5643, 42.0, 'target graphite'),
    ('Cone-droplet study (SPC/E)', 0.5644, 50.7, 'simulated graphene'),
]
lit_rows = [(s, e, a, k, 'lit') for s, e, a, k in lit_rows]

df = pd.DataFrame(own_rows + lit_rows, columns=['source', 'epsilon', 'angle', 'kind', 'group'])
df.to_csv(os.path.join(OUT, 'dataset_combined.csv'), index=False)

eps = df['epsilon'].values
ang = df['angle'].values
own_mask = (df['group'] == 'own').values
lit_mask = ~own_mask
n = len(df)
print(f"Total points: {n} (own: {own_mask.sum()}, literature: {lit_mask.sum()})")


def to_angle(c):
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def norm(tr, te):
    m, s = tr.mean(), tr.std() + 1e-9
    return (tr - m) / s, (te - m) / s


def m_linear(x, y, xt):
    p = np.polyfit(x, np.cos(np.radians(y)), 1)
    return to_angle(np.polyval(p, xt)), None


def m_quad(x, y, xt):
    p = np.polyfit(x, np.cos(np.radians(y)), 2)
    return to_angle(np.polyval(p, xt)), None


def m_xgb(x, y, xt):
    mdl = xgb.XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.1, reg_lambda=1.0, verbosity=0)
    mdl.fit(x.reshape(-1, 1), y)
    return mdl.predict(xt.reshape(-1, 1)), None


def m_gpr(x, y, xt):
    k = C(1.0) * RBF(0.5) + WhiteKernel(0.1)
    g = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=5, normalize_y=True)
    g.fit(x.reshape(-1, 1), y)
    p, s = g.predict(xt.reshape(-1, 1), return_std=True)
    return p, s


def train_net(xn, target, bounded, seed, epochs=500):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.02, weight_decay=1e-3)
    X = torch.tensor(xn, dtype=torch.float32).reshape(-1, 1)
    Y = torch.tensor(target, dtype=torch.float32).reshape(-1, 1)
    for _ in range(epochs):
        opt.zero_grad()
        out = net(X)
        if bounded:
            out = torch.tanh(out)
        loss = ((out - Y) ** 2).mean()
        loss.backward()
        opt.step()
    return net


def predict_net(net, xtn, bounded):
    with torch.no_grad():
        out = net(torch.tensor(xtn, dtype=torch.float32).reshape(-1, 1))
        if bounded:
            out = torch.tanh(out)
    return out.numpy().ravel()


def m_ann(x, y, xt, n_ens=5):
    xn, xtn = norm(x, xt)
    mu, sd = y.mean(), y.std() + 1e-9
    preds = []
    for s in range(n_ens):
        net = train_net(xn, (y - mu) / sd, False, s)
        preds.append(predict_net(net, xtn, False) * sd + mu)
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)


def m_pinn(x, y, xt, n_ens=10):
    xn, xtn = norm(x, xt)
    cy = np.cos(np.radians(y))
    preds = []
    for s in range(n_ens):
        net = train_net(xn, cy, True, s)
        preds.append(to_angle(predict_net(net, xtn, True)))
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)


models = {
    'Linear cos fit': m_linear,
    'Quadratic cos fit': m_quad,
    'XGBoost': m_xgb,
    'GPR': m_gpr,
    'Plain ANN': m_ann,
    'Physics-constrained NN': m_pinn,
}


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


loo_preds = {}
rows = []
for name, fn in models.items():
    print(f"Running {name} ...")
    p = np.zeros(n)
    for tr, te in LeaveOneOut().split(eps):
        p[te] = fn(eps[tr], ang[tr], eps[te])[0]
    loo_preds[name] = p
    p_own_to_lit = fn(eps[own_mask], ang[own_mask], eps[lit_mask])[0]
    p_lit_to_own = fn(eps[lit_mask], ang[lit_mask], eps[own_mask])[0]
    rows.append({
        'Model': name,
        'LOO MAE (all)': mae(ang, p),
        'LOO RMSE (all)': rmse(ang, p),
        'LOO MAE (own 5)': mae(ang[own_mask], p[own_mask]),
        'LOO MAE (literature)': mae(ang[lit_mask], p[lit_mask]),
        'Train own -> test lit MAE': mae(ang[lit_mask], p_own_to_lit),
        'Train lit -> test own MAE': mae(ang[own_mask], p_lit_to_own),
    })

res = pd.DataFrame(rows).round(2)
res.to_csv(os.path.join(OUT, 'model_errors.csv'), index=False)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 20)
print("\nErrors (degrees):")
print(res.to_string(index=False))

pd.DataFrame({'source': df['source'], 'epsilon': eps, 'actual': ang, **loo_preds}).round(2).to_csv(
    os.path.join(OUT, 'loo_predictions.csv'), index=False)

colors = {'own': 'tab:red', 'lit': 'tab:blue'}
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(eps[lit_mask], ang[lit_mask], c='tab:blue', label='Literature', s=50)
ax.scatter(eps[own_mask], ang[own_mask], c='tab:red', label='This work (corrected)', s=70, marker='D')
ax.set_xlabel('epsilon')
ax.set_ylabel('Contact angle (deg)')
ax.set_title('Combined dataset')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, '1_dataset.png'), dpi=150)
plt.close(fig)

grid = np.linspace(0.0, 1.4, 200)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(eps[lit_mask], ang[lit_mask], c='tab:blue', label='Literature', s=40, zorder=5)
ax.scatter(eps[own_mask], ang[own_mask], c='tab:red', label='This work', s=60, marker='D', zorder=5)
for name, fn in models.items():
    p, s = fn(eps, ang, grid)
    line, = ax.plot(grid, p, label=name)
    if s is not None and name in ('GPR', 'Physics-constrained NN'):
        ax.fill_between(grid, p - 2 * s, p + 2 * s, color=line.get_color(), alpha=0.15)
ax.set_ylim(-10, 190)
ax.set_xlabel('epsilon')
ax.set_ylabel('Contact angle (deg)')
ax.set_title('Model fits on all data (bands: +/-2 std for GPR and ensemble NN)')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, '2_model_fits.png'), dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, (name, p) in zip(axes.ravel(), loo_preds.items()):
    ax.scatter(ang[lit_mask], p[lit_mask], c='tab:blue', s=35, label='Literature')
    ax.scatter(ang[own_mask], p[own_mask], c='tab:red', s=55, marker='D', label='This work')
    ax.plot([0, 180], [0, 180], 'k--', lw=1)
    ax.set_xlim(-10, 190)
    ax.set_ylim(-10, 190)
    ax.set_title(f"{name} (MAE {mae(ang, p):.1f})")
    ax.set_xlabel('Actual angle')
    ax.set_ylabel('LOO predicted angle')
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, '3_loo_pred_vs_actual.png'), dpi=150)
plt.close(fig)

cols = ['LOO MAE (all)', 'Train own -> test lit MAE', 'Train lit -> test own MAE']
x = np.arange(len(res))
w = 0.27
fig, ax = plt.subplots(figsize=(11, 5.5))
for i, c in enumerate(cols):
    ax.bar(x + (i - 1) * w, res[c], w, label=c)
ax.set_xticks(x)
ax.set_xticklabels(res['Model'], rotation=20, ha='right')
ax.set_ylabel('MAE (deg)')
ax.set_title('Error comparison')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, '4_error_comparison.png'), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
order = np.argsort(eps)
for name, p in loo_preds.items():
    ax.plot(eps[order], (p - ang)[order], marker='o', ms=4, label=name)
ax.axhline(0, color='k', lw=1)
ax.set_xlabel('epsilon')
ax.set_ylabel('LOO residual (pred - actual, deg)')
ax.set_title('LOO residuals vs epsilon')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, '5_residuals.png'), dpi=150)
plt.close(fig)

print(f"\nSaved CSVs and plots in ./{OUT}/")
