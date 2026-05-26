# -*- coding: utf-8 -*-
"""
泊松过程交互式可视化应用
=========================
满足《概率论与随机过程》期中作业全部要求：
  - 基本性质、叠加性、稀释性、客服中心应用（含 M/M/c 排队论）
  - 理论值与模拟值对比，KS 检验，置信区间
  - 学术严谨 + 交互友好 + 叙事性引导

运行方法：
  pip install streamlit numpy plotly scipy pandas
  streamlit run app.py
"""

import sys
import os

# ── Windows UTF-8 编码强制修复 ──
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream_name in ("stdin", "stdout", "stderr"):
        try:
            getattr(sys, stream_name).reconfigure(encoding="utf-8")
        except Exception:
            pass

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from math import factorial, exp

# ── 页面配置 ─────────────────────────────────────────────
st.set_page_config(page_title="泊松过程交互式可视化", layout="wide")

# ── 禁止浏览器自动翻译 + MathJax ──
st.markdown("""
<script>
document.documentElement.classList.add('notranslate');
document.documentElement.lang = 'zh-CN';
</script>
""", unsafe_allow_html=True)

# ── 自定义 CSS ───────────────────────────────────────────
st.markdown("""
<style>
    /* ================================================================
       Design System — 简约风格，高可视性
       ================================================================ */
    :root {
        --c-primary: #6366f1;
        --c-primary-lt: #eef2ff;
        --c-success: #10b981;
        --c-success-lt: #ecfdf5;
        --c-warning: #f59e0b;
        --c-warning-lt: #fffbeb;
        --c-danger: #ef4444;
        --c-danger-lt: #fef2f2;
        --c-slate-50: #f8fafc;
        --c-slate-100: #f1f5f9;
        --c-slate-200: #e2e8f0;
        --c-slate-300: #cbd5e1;
        --c-slate-400: #94a3b8;
        --c-slate-500: #64748b;
        --c-slate-600: #475569;
        --c-slate-700: #334155;
        --c-slate-800: #1e293b;
        --c-slate-900: #0f172a;
        --radius-lg: 16px;
        --radius: 12px;
        --radius-sm: 8px;
        --shadow-sm: 0 1px 2px rgba(0,0,0,.04);
        --shadow: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
        --shadow-md: 0 4px 6px rgba(0,0,0,.05), 0 2px 4px rgba(0,0,0,.04);
        --shadow-lg: 0 10px 25px rgba(0,0,0,.08), 0 4px 10px rgba(0,0,0,.04);
        --font-sans: 'Microsoft YaHei','PingFang SC','Hiragino Sans GB','Noto Sans SC','Source Han Sans SC','SimHei',sans-serif;
        --font-mono: 'JetBrains Mono','Cascadia Code','Fira Code','Consolas','Microsoft YaHei',monospace;
    }

    /* ===== 全局 ===== */
    *, *::before, *::after {
        font-family: var(--font-sans) !important;
        box-sizing: border-box;
    }
    code, pre, kbd, samp, .stCodeBlock, .stCodeBlock * {
        font-family: var(--font-mono) !important;
    }
    hr { border-color: var(--c-slate-200) !important; margin: 1.5rem 0 !important; }

    /* ===== 隐藏所有 Material Icons（避免乱码） ===== */
    .material-icons, .material-symbols-outlined,
    .material-symbols-rounded, .material-symbols-sharp {
        display: none !important;
    }

    /* ===== Streamlit 容器定制 ===== */
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1280px; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-right: 1px solid var(--c-slate-200);
    }
    [data-testid="stSidebar"] h2 { font-size: 1.2rem !important; font-weight: 700 !important; color: var(--c-slate-800) !important; }
    [data-testid="stSidebar"] h3 { font-size: 0.95rem !important; font-weight: 600 !important; color: var(--c-slate-700) !important; }
    .stButton button {
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        border: 1px solid var(--c-slate-300) !important;
        background: #fff !important;
        color: var(--c-slate-700) !important;
        transition: all .15s !important;
    }
    .stButton button:hover { border-color: var(--c-primary) !important; color: var(--c-primary) !important; background: var(--c-primary-lt) !important; }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600 !important; font-size: 0.92rem !important;
        padding: 10px 20px !important; border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--c-primary) !important;
        border-bottom: 2px solid var(--c-primary) !important;
    }

    /* ===== 指标卡片 ===== */
    .metric-card {
        background: #fff;
        border: 1px solid var(--c-slate-200);
        border-radius: var(--radius);
        padding: 22px 20px;
        text-align: center;
        box-shadow: var(--shadow-sm);
        transition: box-shadow .2s, transform .2s;
    }
    .metric-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--c-slate-800);
        line-height: 1.15;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 0.8rem;
        color: var(--c-slate-500);
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 500;
    }
    .metric-delta {
        font-size: 0.78rem;
        margin-top: 3px;
        color: var(--c-slate-400);
    }
    .metric-green  { color: var(--c-success); font-weight: 700; }
    .metric-yellow { color: var(--c-warning); font-weight: 700; }
    .metric-red    { color: var(--c-danger);  font-weight: 700; }

    /* ===== 理论/结论/提示框 ===== */
    .theory-box, .conclusion-box, .highlight-box {
        border-radius: 0 var(--radius) var(--radius) 0;
        padding: 18px 22px;
        margin: 18px 0;
        font-size: 0.9rem;
        line-height: 1.8;
        box-shadow: var(--shadow-sm);
    }
    .theory-box {
        background: linear-gradient(135deg, var(--c-primary-lt), #f5f3ff 100%);
        border-left: 4px solid var(--c-primary);
    }
    .conclusion-box {
        background: linear-gradient(135deg, var(--c-success-lt), #f0fdf4 100%);
        border-left: 4px solid var(--c-success);
    }
    .highlight-box {
        background: linear-gradient(135deg, var(--c-warning-lt), #fff7ed 100%);
        border-left: 4px solid var(--c-warning);
    }

    /* ===== Hero ===== */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 25%, #3730a3 60%, #0f172a 100%);
        border-radius: 20px;
        padding: 52px 60px;
        margin: 0 0 36px 0;
        color: #f1f5f9;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-lg);
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -30%; left: -30%;
        width: 160%; height: 160%;
        background:
            radial-gradient(circle at 25% 60%, rgba(99,102,241,.15) 0%, transparent 50%),
            radial-gradient(circle at 70% 40%, rgba(139,92,246,.10) 0%, transparent 50%),
            radial-gradient(circle at 50% 80%, rgba(59,130,246,.08) 0%, transparent 40%);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 14px;
        position: relative;
        background: linear-gradient(135deg, #a5b4fc, #c4b5fd, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #94a3b8;
        line-height: 1.75;
        position: relative;
        max-width: 700px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99,102,241,.16);
        border: 1px solid rgba(99,102,241,.28);
        border-radius: 20px;
        padding: 5px 16px;
        font-size: 0.85rem;
        color: #a5b4fc;
        margin-right: 10px;
        margin-bottom: 8px;
        position: relative;
        font-weight: 500;
        backdrop-filter: blur(4px);
    }

    /* ===== 步骤卡片 ===== */
    .step-row {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin: 28px 0;
    }
    .step-card {
        flex: 1;
        min-width: 150px;
        background: #fff;
        border: 1px solid var(--c-slate-200);
        border-radius: var(--radius);
        padding: 22px 18px;
        text-align: center;
        box-shadow: var(--shadow-sm);
        transition: box-shadow .2s, transform .2s;
    }
    .step-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
        border-color: var(--c-primary);
    }
    .step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px; height: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--c-primary), #8b5cf6);
        color: #fff;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 12px;
        box-shadow: 0 4px 10px rgba(99,102,241,.25);
    }
    .step-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: var(--c-slate-800);
        margin-bottom: 6px;
    }
    .step-desc {
        font-size: 0.8rem;
        color: var(--c-slate-500);
        line-height: 1.6;
    }

    /* ===== 响应式 ===== */
    @media (max-width: 768px) {
        .hero-container { padding: 32px 28px; }
        .hero-title { font-size: 1.7rem; }
        .step-row { flex-direction: column; }
        .metric-value { font-size: 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 工具函数
# ============================================================

def generate_poisson_process(lam, T):
    """生成泊松过程的一次样本路径，返回 (到达时刻数组, 间隔时间数组)"""
    expected = max(int(lam * T * 1.5) + 10, 10)
    interarrival = np.random.exponential(1.0 / lam, size=expected)
    arrival_times = np.cumsum(interarrival)
    mask = arrival_times < T
    arrival_times = arrival_times[mask]
    interarrival = interarrival[mask]
    return arrival_times, interarrival


def make_step_xy(arrival_times, T):
    """将到达时刻转换为阶梯图的 (x, y) 坐标"""
    x, y = [0.0], [0]
    for i, t in enumerate(arrival_times):
        x.extend([t, t])
        y.extend([i, i + 1])
    x.append(T)
    y.append(len(arrival_times))
    return x, y


def get_error_color(rel_error):
    """相对误差着色：绿 <5%，黄 5-10%，红 >10%"""
    if rel_error < 0.05:
        return "#16a34a", "metric-green"
    elif rel_error < 0.10:
        return "#d97706", "metric-yellow"
    else:
        return "#dc2626", "metric-red"


def compute_r_squared(observed, expected):
    """R²"""
    ss_res = np.sum((observed - expected) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1 - ss_res / ss_tot


def format_error_html(sim_val, theory_val):
    """生成带颜色标注的误差 HTML"""
    if theory_val == 0:
        rel_err = abs(sim_val) if sim_val != 0 else 0
    else:
        rel_err = abs(sim_val - theory_val) / abs(theory_val)
    color_hex, color_class = get_error_color(rel_err)
    html = (
        f'<span style="color:{color_hex};font-weight:bold">{sim_val:.4f}</span> '
        f'(理论 {theory_val:.4f}，误差 <b>{rel_err:.2%}</b>)'
    )
    return html, rel_err


def collect_interarrivals(lam, T, n_runs):
    """多次生成泊松过程并汇总所有间隔时间"""
    all_interarrivals = []
    for _ in range(n_runs):
        _, ia = generate_poisson_process(lam, T)
        all_interarrivals.extend(ia.tolist())
    return np.array(all_interarrivals)


def plot_interarrival_histogram(interarrivals, lam, title="到达间隔时间分布"):
    """间隔时间直方图 + 理论指数分布 PDF，返回 (fig, R²)"""
    if len(interarrivals) < 5:
        return None, None

    nbins = min(50, max(10, int(np.sqrt(len(interarrivals)))))
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=interarrivals, histnorm='probability density',
        nbinsx=nbins,
        name='模拟数据', marker_color='#3b82f6', opacity=0.65,
        hovertemplate='间隔: %{x:.3f}<br>密度: %{y:.4f}<extra></extra>'
    ))

    x_max = np.max(interarrivals) * 1.1
    x_theory = np.linspace(0, x_max, 200)
    pdf_theory = lam * np.exp(-lam * x_theory)
    fig.add_trace(go.Scatter(
        x=x_theory, y=pdf_theory, mode='lines',
        name=f'理论 Exp(λ={lam})',
        line=dict(color='#ef4444', width=2.5, dash='dash'),
        hovertemplate='x: %{x:.3f}<br>f(x): %{y:.4f}<extra></extra>'
    ))

    counts, bin_edges = np.histogram(interarrivals, bins=nbins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    expected_pdf = lam * np.exp(-lam * bin_centers)
    r2 = compute_r_squared(counts, expected_pdf)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title=dict(text='间隔时间', font=dict(size=14)),
        yaxis_title=dict(text='概率密度', font=dict(size=14)),
        template='simple_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
    return fig, r2


def show_error_table(results):
    """显示理论值 vs 模拟值对比表格"""
    cols = st.columns(4)
    cols[0].markdown("**指标**")
    cols[1].markdown("**理论值**")
    cols[2].markdown("**模拟值**")
    cols[3].markdown("**相对误差**")
    for name, theory, sim in results:
        html_err, rel_err = format_error_html(sim, theory)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(name)
        c2.markdown(f"{theory:.4f}")
        c3.markdown(f"{sim:.4f}" if isinstance(sim, (int, float)) else str(sim))
        c4.markdown(
            f'<span class="{get_error_color(rel_err)[1]}">{rel_err:.2%}</span>',
            unsafe_allow_html=True
        )


def ks_test_exponential(data, lam):
    """KS 检验：数据是否服从 Exp(λ) 分布。返回 (D值, p值, 结论文本)"""
    if len(data) < 5:
        return None, None, "样本过少，无法检验"
    d_stat, p_value = stats.kstest(data, 'expon', args=(0, 1.0 / lam))
    if p_value >= 0.05:
        conclusion = f"p={p_value:.4f} >= 0.05，不拒绝 H0，数据与 Exp(λ={lam}) 无显著差异"
        color = "#16a34a"
    elif p_value >= 0.01:
        conclusion = f"p={p_value:.4f} < 0.05，在 5% 水平拒绝 H0，差异显著"
        color = "#d97706"
    else:
        conclusion = f"p={p_value:.4f} < 0.01，在 1% 水平拒绝 H0，差异高度显著"
        color = "#dc2626"
    return d_stat, p_value, f'<span style="color:{color}">{conclusion}</span>'


def erlang_c(lam, mu, c):
    """
    Erlang C 公式
    参数: lam = 到达率, mu = 服务率（每坐席）, c = 坐席数
    返回: (P_wait, Lq, Wq, rho) - 等待概率、平均队长、平均等待时间、利用率
    """
    rho = lam / (c * mu)
    if rho >= 1.0:
        return 1.0, float('inf'), float('inf'), rho

    a = lam / mu
    sum_term = sum(a**k / factorial(k) for k in range(c))
    last_term = (a**c / factorial(c)) * (1 / (1 - rho))
    p0 = 1.0 / (sum_term + last_term)
    p_wait = (a**c / factorial(c)) * (c / (c - a)) * p0

    try:
        lq = p_wait * rho / (1 - rho)
        wq = lq / lam
    except (ZeroDivisionError, OverflowError):
        lq, wq = float('inf'), float('inf')

    return p_wait, lq, wq, rho


def simulate_mmc_queue(lam, mu, c, T):
    """
    离散事件模拟 M/M/c 排队系统
    返回 dict: arrivals, waits, queue_len_over_time, system_len_over_time, 等
    """
    expected_n = max(int(lam * T * 1.5) + 100, 100)
    interarrivals = np.random.exponential(1.0 / lam, size=expected_n)
    arrival_times = np.cumsum(interarrivals)
    mask = arrival_times < T
    arrival_times = arrival_times[mask]
    n_total = len(arrival_times)

    empty_result = {
        'arrivals': np.array([]), 'service_start': np.array([]),
        'departures': np.array([]), 'waits': np.array([]),
        'queue_len_over_time': ([0, T], [0, 0]),
        'system_len_over_time': ([0, T], [0, 0]),
        'n_total': 0, 'n_served': 0, 'avg_wait': 0, 'max_queue': 0,
    }

    if n_total == 0:
        return empty_result

    service_times = np.random.exponential(1.0 / mu, size=n_total)
    service_start = np.zeros(n_total)
    departures = np.zeros(n_total)
    waits = np.zeros(n_total)
    server_free_time = np.zeros(c)

    for i in range(n_total):
        at = arrival_times[i]
        earliest_free = np.min(server_free_time)
        assigned_server = np.argmin(server_free_time)
        start_time = max(at, earliest_free)
        service_start[i] = start_time
        waits[i] = start_time - at
        departures[i] = start_time + service_times[i]
        server_free_time[assigned_server] = departures[i]

    # 构建队列长度时间序列：扫描所有到达/离开事件
    events = [(0.0, 0)]  # (time, net_delta: +1 for arrival, -1 for departure)
    for i in range(n_total):
        events.append((arrival_times[i], +1))
        events.append((departures[i], -1))
    events.append((T, 0))
    events.sort(key=lambda x: x[0])

    rec_t, rec_q, rec_s = [], [], []
    n_in_system = 0
    for et, delta in events:
        rec_t.append(et)
        n_in_system += delta
        in_queue = max(0, n_in_system - c)
        rec_q.append(in_queue)
        rec_s.append(n_in_system)

    # 去重同一时间点
    seen = {}
    for t_val, q_val, s_val in zip(rec_t, rec_q, rec_s):
        seen[t_val] = (q_val, s_val)
    rec_t = np.array(sorted(seen.keys()))
    rec_q = np.array([seen[t][0] for t in rec_t])
    rec_s = np.array([seen[t][1] for t in rec_t])

    avg_wait = np.mean(waits)
    max_queue = int(max(rec_q))

    return {
        'arrivals': arrival_times,
        'service_start': service_start,
        'departures': departures,
        'waits': waits,
        'queue_len_over_time': (rec_t, rec_q),
        'system_len_over_time': (rec_t, rec_s),
        'n_total': n_total,
        'n_served': n_total,
        'avg_wait': avg_wait,
        'max_queue': max_queue,
    }


# ============================================================
# 首页总览
# ============================================================

def render_tab_home():
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">泊松过程交互式可视化</div>
        <div class="hero-subtitle">
            从数学定义出发，逐步探索泊松过程的<b>基本性质</b>、<b>叠加性</b>与<b>稀释性</b>，
            最终在<b>客服中心排队系统</b>中实战应用 M/M/c 排队论 -
            每个环节均可通过滑块、按钮实时操控参数，即时观察理论值与模拟值的对比。
        </div>
        <div style="margin-top:18px; position:relative;">
            <span class="hero-badge">泊松过程</span>
            <span class="hero-badge">指数分布</span>
            <span class="hero-badge">M/M/c 排队论</span>
            <span class="hero-badge">Erlang C</span>
            <span class="hero-badge">离散事件模拟</span>
            <span class="hero-badge">KS 检验</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 学习路径
    st.markdown("### 探索路径")
    st.markdown("""
    <div class="step-row">
        <div class="step-card">
            <div class="step-num">1</div>
            <div class="step-title">基本性质</div>
            <div class="step-desc">理解泊松过程定义<br>验证到达间隔的指数分布<br>探究无记忆性</div>
        </div>
        <div class="step-card">
            <div class="step-num">2</div>
            <div class="step-title">叠加性</div>
            <div class="step-desc">两条独立泊松流的合并<br>验证强度相加 入=入1+入2<br>排队论中多源合并的理论基础</div>
        </div>
        <div class="step-card">
            <div class="step-num">3</div>
            <div class="step-title">稀释性</div>
            <div class="step-desc">Bernoulli 随机筛选<br>验证强度折减 入'=入p<br>理解丢包/过滤模型</div>
        </div>
        <div class="step-card">
            <div class="step-num">4</div>
            <div class="step-title">客服中心实战</div>
            <div class="step-desc">多线路来电叠加<br>骚扰电话过滤<br>M/M/c 排队 + Erlang C<br>坐席资源优化</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 关键概念速览
    st.markdown("### 泊松过程速览")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="theory-box">
        <b>定义（四条公理）</b><br>
        计数过程 {N(t), t >= 0} 称为强度为 入 的<em>泊松过程</em>，若：<br>
        (1) N(0)=0<br>
        (2) 独立增量：不重叠区间内的事件数独立<br>
        (3) 平稳增量：分布仅依赖于区间长度<br>
        (4) P(N(h)=1)=入h+o(h)，P(N(h)>=2)=o(h)
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="conclusion-box">
        <b>核心性质</b><br>
        - N(t) ~ Poisson(入t)<br>
        - E[N(t)] = Var[N(t)] = 入t<br>
        - 到达间隔 Ti ~ Exp(入)，i.i.d.<br>
        - 无记忆性：P(T>s+t | T>t)=P(T>s)<br>
        - 叠加：独立泊松过程之和仍为泊松过程<br>
        - 稀释：Bernoulli 筛选后仍为泊松过程
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight-box">
    <b>使用提示</b>：通过左侧控制面板调整参数，每个 Tab 页提供：
    理论公式 → 样本路径可视化 → 统计分布验证 → 模拟与理论对比 → 问题思考与结论。<br>
    <b>推荐先阅读每个 Tab 顶部的理论框</b>，理解数学背景后再动手实验。
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 标签页 1：泊松过程基本性质
# ============================================================

def render_tab_basic(lam, T, n_samples):
    st.markdown("## 泊松过程基本性质")

    st.markdown(f"""
    <div class="theory-box">
    <b>定义</b>：计数过程 {{N(t), t ≥ 0}} 称为强度为 λ 的<em>泊松过程</em>，若满足：<br>
    ① N(0)=0；② 独立增量；③ 平稳增量；
    ④ P(N(h)=1) = λh + o(h)，P(N(h) ≥ 2) = o(h)。<br><br>
    <b>物理意义</b>：λ={lam} 表示单位时间内事件发生的平均次数（强度）。
    间隔时间独立同分布于指数分布 Exp(λ={lam})。
    期望事件数 E[N(t)] = λt = {lam}t，方差 Var[N(t)] = λt。<br><br>
    <b>通俗理解</b>：想象一个公交站台，如果公交车按泊松过程到站，λ=2 表示平均每小时来 2 辆车。
    到站间隔服从指数分布——短间隔常见，长间隔稀少。而且"已经等了 10 分钟"不会改变"还要等多久"的分布（无记忆性）。
    </div>
    """, unsafe_allow_html=True)

    # ── 样本路径 ──
    st.markdown("### 样本路径可视化")
    st.markdown("""
    <div class="highlight-box" style="font-size:0.85rem;">
    <b>如何阅读此图：</b> 横轴为时间 t，纵轴为累计事件数 N(t)。每条阶梯线代表一次独立的随机实验。
    线越陡说明该时段内事件越密集。注意观察：① 跳跃高度始终为 1（任意时刻最多发生一个事件）；
    ② 不同路径虽然形状各异，但事件总数的波动范围与理论值 λT 一致。<br>
    <b>试试：</b> 将 λ 从 1.0 切换到 5.0，观察路径的密集程度变化。
    </div>
    """, unsafe_allow_html=True)
    colors = ['#3b82f6', '#ef4444', '#22c55e']

    arrivals_list = []
    for _ in range(3):
        at, _ = generate_poisson_process(lam, T)
        arrivals_list.append(at)

    fig_paths = go.Figure()
    for i, at in enumerate(arrivals_list):
        sx, sy = make_step_xy(at, T)
        fig_paths.add_trace(go.Scatter(
            x=sx, y=sy, mode='lines',
            name=f'路径 {i+1} (N(T)={len(at)}, lambda_hat={len(at)/T:.2f})',
            line=dict(color=colors[i], width=2.2, shape='hv'),
            hovertemplate='t: %{x:.2f}<br>N(t): %{y}<extra></extra>'
        ))
        if len(at) > 0:
            fig_paths.add_trace(go.Scatter(
                x=at, y=list(range(1, len(at) + 1)),
                mode='markers',
                marker=dict(size=5, color=colors[i], line=dict(width=0)),
                showlegend=False,
                hovertemplate='到达时间: %{x:.3f}<br>事件编号: %{y}<extra></extra>'
            ))

    fig_paths.update_layout(
        title=dict(text=f"泊松过程样本路径 (lambda={lam})", font=dict(size=16)),
        xaxis_title=dict(text='时间 t', font=dict(size=14)),
        yaxis_title=dict(text='N(t)', font=dict(size=14)),
        template='simple_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    fig_paths.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
    fig_paths.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
    st.plotly_chart(fig_paths, use_container_width=True)

    # ── 增量分布 ──
    st.markdown("### 固定区间内事件数的分布")
    st.markdown("""
    <div class="highlight-box" style="font-size:0.85rem;">
    <b>如何阅读此图：</b> 蓝色柱状图为模拟频率，红色虚线为理论 Poisson(λT) 的 PMF。
    两者越吻合，说明模拟越成功地复现了泊松过程的增量分布性质。观察柱状图的中心位置——它应该在 λT 附近。<br>
    <b>关键概念：</b> 这是泊松过程最核心的性质：N(t) ~ Poisson(λt)。在任意固定长度区间内，事件数服从泊松分布。
    </div>
    """, unsafe_allow_html=True)
    n_runs = max(1, n_samples // 10)
    counts_T = []
    for _ in range(n_runs):
        at, _ = generate_poisson_process(lam, T)
        counts_T.append(len(at))
    counts_T = np.array(counts_T)

    fig_pois = go.Figure()
    max_k = max(np.max(counts_T), 10)
    k_range = np.arange(0, max_k + 1)
    pmf_theory = stats.poisson.pmf(k_range, lam * T)
    hist_k = np.bincount(counts_T, minlength=max_k + 1) / len(counts_T)

    fig_pois.add_trace(go.Bar(
        x=k_range, y=hist_k,
        name=f'模拟 (n={n_runs}次)',
        marker_color='#3b82f6', opacity=0.65,
        hovertemplate='k: %{x}<br>频率: %{y:.4f}<extra></extra>'
    ))
    fig_pois.add_trace(go.Scatter(
        x=k_range, y=pmf_theory, mode='markers+lines',
        name=f'理论 Poisson(lambdaT={lam*T:.1f})',
        marker=dict(color='#ef4444', size=6), line=dict(color='#ef4444', width=2, dash='dash'),
        hovertemplate='k: %{x}<br>P(N(T)=k): %{y:.4f}<extra></extra>'
    ))

    fig_pois.update_layout(
        title=dict(text=f"区间 [0,{T}] 内事件数分布 (理论 lambdaT={lam*T:.1f})", font=dict(size=16)),
        xaxis_title=dict(text='事件数 k', font=dict(size=14)),
        yaxis_title=dict(text='概率 / 频率', font=dict(size=14)),
        template='simple_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=20, t=50, b=40), bargap=0.15
    )
    fig_pois.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
    fig_pois.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
    st.plotly_chart(fig_pois, use_container_width=True)

    # ── 间隔时间分析 ──
    st.markdown("### 到达间隔时间分析")
    st.markdown("""
    <div class="highlight-box" style="font-size:0.85rem;">
    <b>如何阅读此图：</b> 蓝色直方图为模拟生成的所有到达间隔时间的经验分布，红色虚线为理论 Exp(λ) 的 PDF。
    注意指数分布的典型形态——单调递减，短间隔概率高，长间隔概率低。R^2 越接近 1 拟合越好。<br>
    <b>KS 检验说明：</b> Kolmogorov-Smirnov 检验比较经验分布与理论分布的最大偏差 D。p ≥ 0.05 意味着"数据与理论无显著差异"（这是我们期望的结果）。
    </div>
    """, unsafe_allow_html=True)
    all_ia = collect_interarrivals(lam, T, n_runs)
    fig_hist, r2 = plot_interarrival_histogram(all_ia, lam,
        title=f"到达间隔时间分布 (lambda={lam}, 样本数={len(all_ia)})")
    if fig_hist is not None:
        st.plotly_chart(fig_hist, use_container_width=True)

    # 统计指标 + KS 检验
    col_a, col_b, col_c = st.columns(3)
    sim_mean = np.mean(all_ia)
    theory_mean = 1.0 / lam
    html_err, rel_err = format_error_html(sim_mean, theory_mean)
    col_a.metric("模拟均值", f"{sim_mean:.4f}", f"理论 {theory_mean:.4f}")
    col_b.metric("理论均值 1/lambda", f"{theory_mean:.4f}")

    d_stat, p_val, ks_conclusion = ks_test_exponential(all_ia, lam)
    if d_stat is not None:
        col_c.markdown(
            f"**KS 检验**: D={d_stat:.4f}<br>{ks_conclusion}",
            unsafe_allow_html=True
        )
        st.markdown(
            f"**拟合优度 R^2** = {r2:.4f}  |  D={d_stat:.4f}, p={p_val:.4f}"
            if r2 is not None else
            f"**KS 检验**: D={d_stat:.4f}, p={p_val:.4f}"
        )

    # ── 无记忆性验证 ──
    st.markdown("### 无记忆性验证")
    st.markdown("""
    <div class="highlight-box" style="font-size:0.85rem;">
    <b>无记忆性是什么？</b> 指数分布是唯一具有无记忆性的连续分布：无论已经等待了多久，剩余等待时间的分布始终不变。
    这意味着"过去"对未来没有任何影响——系统永远"刷新"到初始状态。<br>
    <b>如何验证：</b> 蓝色为全部间隔的分布，橙色为"已等待 t₀ 后剩余时间"的分布，红色为理论 Exp(λ)。
    如果三者重合，则无记忆性成立。下方均值对比也应当接近（差值越小越验证）。
    </div>
    """, unsafe_allow_html=True)

    col_t0, _, _ = st.columns([1, 1, 3])
    with col_t0:
        t0 = st.number_input(
            "条件时间 t0", value=5.0, min_value=0.1, max_value=50.0, step=0.5,
            key="basic_t0"
        )

    n_verify = 3000
    X = np.random.exponential(1.0 / lam, size=n_verify)
    mask = X > t0
    remaining = X[mask] - t0

    fig_memory = go.Figure()
    bins_mem = min(50, max(10, int(np.sqrt(n_verify))))

    fig_memory.add_trace(go.Histogram(
        x=X, histnorm='probability density', nbinsx=bins_mem,
        name=f'全部间隔时间 (n={n_verify})',
        marker_color='#3b82f6', opacity=0.55,
        hovertemplate='间隔: %{x:.3f}<br>密度: %{y:.4f}<extra></extra>'
    ))
    fig_memory.add_trace(go.Histogram(
        x=remaining, histnorm='probability density', nbinsx=bins_mem,
        name=f't0={t0} 后剩余时间 (n={len(remaining)})',
        marker_color='#f59e0b', opacity=0.55,
        hovertemplate='剩余时间: %{x:.3f}<br>密度: %{y:.4f}<extra></extra>'
    ))
    x_max = max(np.max(X), np.max(remaining) if len(remaining) > 0 else 0) * 1.1
    x_theory = np.linspace(0, max(x_max, 0.1), 300)
    pdf_theory = lam * np.exp(-lam * x_theory)
    fig_memory.add_trace(go.Scatter(
        x=x_theory, y=pdf_theory, mode='lines',
        name=f'理论 Exp(lambda={lam})',
        line=dict(color='#ef4444', width=2.5, dash='dash'),
        hovertemplate='x: %{x:.3f}<br>f(x): %{y:.4f}<extra></extra>'
    ))

    fig_memory.update_layout(
        title=dict(text=f"无记忆性验证 (lambda={lam}, t0={t0})", font=dict(size=16)),
        xaxis_title=dict(text='间隔时间', font=dict(size=14)),
        yaxis_title=dict(text='概率密度', font=dict(size=14)),
        template='simple_white', hovermode='x unified',
        barmode='overlay',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    fig_memory.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
    fig_memory.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
    st.plotly_chart(fig_memory, use_container_width=True)

    c1, c2 = st.columns(2)
    mean_all = np.mean(X)
    mean_rem = np.mean(remaining) if len(remaining) > 0 else 0
    c1.metric("全部间隔时间均值", f"{mean_all:.4f}", f"理论 1/lambda={theory_mean:.4f}")
    c2.metric("t0 后剩余时间均值", f"{mean_rem:.4f}", f"理论 1/lambda={theory_mean:.4f}")
    diff = abs(mean_all - mean_rem)
    st.markdown(f"**两均值之差的绝对值** = {diff:.4f}（越小越验证无记忆性）")

    st.markdown("""
    <div class="conclusion-box">
    <b>问题思考与结论</b><br>
    ① <b>到达间隔为何服从指数分布？</b> 由泊松过程平稳独立增量性质可推导间隔时间 T_i 满足
    P(T_1 > t) = P(N(t)=0) = e^{-lambda t}，故 T_i ~ Exp(lambda)。这并非巧合，而是公理定义的必然推论。<br>
    ② <b>KS 检验的意义：</b> KS 统计量 D 衡量经验分布与理论分布的最大偏差，p 值越大说明拟合越好。
    统计学中 p >= 0.05 通常作为"不拒绝原假设"的阈值。<br>
    ③ <b>无记忆性的实际意义：</b> 无论系统已运行多久，下一事件的等待时间分布始终不变，这极大简化了排队系统的分析——
    我们不需要记录每个顾客"已经等了多久"，因为剩余等待的分布与刚到达时完全相同。<br>
    ④ <b>样本路径特征：</b> 观察阶梯图可知 N(t) 单调不减、跃度恒为 1（几乎处处），符合泊松过程定义。
    跃度为 1 意味着两个事件不会"同时发生"——这是公理 ④ 的体现。<br>
    ⑤ <b>动手实验建议：</b> 尝试 λ=0.5 和 λ=5.0，对比样本路径的稀疏与密集。观察事件数分布的均值是否从 λT=0.5T 移到了 5T。
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 标签页 2：泊松过程叠加性
# ============================================================

def render_tab_superposition(lam1, lam2, T_sup):
    st.markdown("## 泊松过程叠加性")

    st.markdown(f"""
    <div class="theory-box">
    <b>叠加定理</b>：设 N1(t) 和 N2(t) 为两个<em>独立</em>的泊松过程，
    强度分别为 lambda_1={lam1}, lambda_2={lam2}，
    则叠加过程 N(t) = N1(t) + N2(t) 仍为泊松过程，强度为
    lambda = lambda_1 + lambda_2 = {lam1+lam2}。<br>
    <b>物理意义</b>：多条独立泊松事件流的合并仍为泊松流，强度为各分流强度之和。
    这是排队论中多源输入合并的理论基础。好比两个互不干扰的水龙头同时往一个桶里滴水——合并后的水滴序列仍满足泊松过程的统计规律。<br>
    <b>数学直觉</b>：泊松过程叠加后"仍是自己"——这是泊松过程区别于一般随机过程的一个优良代数封闭性。
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight-box" style="font-size:0.85rem;">
    <b>实验指南：</b> 上图为过程 1 和过程 2 各自的事件计数，下图为合并后的事件计数。
    验证合并后的总事件数是否约为 (λ₁+λ₂)·T。尝试让 λ₁ 远小于 λ₂（如 λ₁=0.2, λ₂=4.0），
    观察叠加过程主要由哪个分过程"贡献"事件。<br>
    <b>关键观察：</b> 叠加过程仍然是泊松过程——事件数服从 Poisson((λ₁+λ₂)T)，间隔时间服从 Exp(λ₁+λ₂)。
    </div>
    """, unsafe_allow_html=True)

    arrivals1, _ = generate_poisson_process(lam1, T_sup)
    arrivals2, _ = generate_poisson_process(lam2, T_sup)
    arrivals_super = np.sort(np.concatenate([arrivals1, arrivals2]))

    st.markdown("### 过程对比")

    fig_super = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f'过程 1 (lambda_1={lam1})，事件数={len(arrivals1)}',
            f'过程 2 (lambda_2={lam2})，事件数={len(arrivals2)}',
            f'叠加过程 (lambda_1+lambda_2={lam1+lam2})，事件数={len(arrivals_super)}'
        )
    )

    data_sets = [
        (arrivals1, '#3b82f6', '过程 1'),
        (arrivals2, '#ef4444', '过程 2'),
        (arrivals_super, '#22c55e', '叠加过程'),
    ]

    for row, (at, color, name) in enumerate(data_sets, start=1):
        sx, sy = make_step_xy(at, T_sup)
        fig_super.add_trace(go.Scatter(
            x=sx, y=sy, mode='lines', name=name,
            line=dict(color=color, width=2.2, shape='hv'),
            showlegend=False,
            hovertemplate='t: %{x:.2f}<br>N(t): %{y}<extra></extra>'
        ), row=row, col=1)

    fig_super.update_xaxes(title_text='时间 t', row=3, col=1)
    fig_super.update_yaxes(title_text='N1(t)', row=1, col=1)
    fig_super.update_yaxes(title_text='N2(t)', row=2, col=1)
    fig_super.update_yaxes(title_text='N(t)', row=3, col=1)
    fig_super.update_layout(
        height=650, template='simple_white', hovermode='x unified',
        title=dict(text="独立泊松过程与其叠加过程对比", font=dict(size=16)),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    for i in range(1, 4):
        fig_super.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9', row=i, col=1)
        fig_super.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9', row=i, col=1)
    st.plotly_chart(fig_super, use_container_width=True)

    # 结果验证表格
    st.markdown("### 理论值与模拟值对比")
    est_lam1 = len(arrivals1) / T_sup
    est_lam2 = len(arrivals2) / T_sup
    est_lam_super = len(arrivals_super) / T_sup

    results = [
        ("过程 1 强度", lam1, est_lam1),
        ("过程 2 强度", lam2, est_lam2),
        ("叠加后强度", lam1 + lam2, est_lam_super),
    ]
    show_error_table(results)

    # 叠加过程统计验证
    st.markdown("### 叠加过程事件数分布验证")
    st.markdown("""
    <div class="highlight-box" style="font-size:0.85rem;">
    <b>如何验证叠加后仍是泊松过程？</b> 系统自动重复生成 500 次叠加过程，
    统计事件数分布并与 Poisson((λ₁+λ₂)T) 的理论 PMF 对比。同时进行卡方拟合优度检验和间隔分布验证。<br>
    <b>三个层次</b>：① 事件数分布 → 卡方检验；② 间隔分布 → KS 检验；③ 间隔均值 → 是否接近 1/(λ₁+λ₂)。
    </div>
    """, unsafe_allow_html=True)
    n_sup_runs = 500
    sup_counts = []
    for _ in range(n_sup_runs):
        a1, _ = generate_poisson_process(lam1, T_sup)
        a2, _ = generate_poisson_process(lam2, T_sup)
        sup_counts.append(len(a1) + len(a2))
    sup_counts = np.array(sup_counts)

    theory_lam = lam1 + lam2
    max_k = max(int(theory_lam * T_sup * 2), np.max(sup_counts), 10)
    k_range = np.arange(0, max_k + 1)

    hist_sup = np.bincount(sup_counts, minlength=max_k + 1)[:max_k + 1] / n_sup_runs
    pmf_theory = stats.poisson.pmf(k_range, theory_lam * T_sup)

    fig_sup_dist = go.Figure()
    fig_sup_dist.add_trace(go.Bar(
        x=k_range, y=hist_sup,
        name=f'模拟 (n={n_sup_runs})',
        marker_color='#3b82f6', opacity=0.65,
        hovertemplate='k: %{x}<br>频率: %{y:.4f}<extra></extra>'
    ))
    fig_sup_dist.add_trace(go.Scatter(
        x=k_range, y=pmf_theory, mode='markers+lines',
        name=f'理论 Poisson({theory_lam*T_sup:.1f})',
        marker=dict(color='#ef4444', size=6), line=dict(color='#ef4444', width=2, dash='dash'),
        hovertemplate='k: %{x}<br>P: %{y:.4f}<extra></extra>'
    ))
    fig_sup_dist.update_layout(
        title=dict(text=f"叠加过程事件数分布 vs Poisson({theory_lam*T_sup:.1f})", font=dict(size=16)),
        xaxis_title=dict(text='事件数 k', font=dict(size=14)),
        yaxis_title=dict(text='概率 / 频率', font=dict(size=14)),
        template='simple_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=20, t=50, b=40), bargap=0.15
    )
    st.plotly_chart(fig_sup_dist, use_container_width=True)

    # 卡方拟合优度
    obs_full = np.bincount(sup_counts, minlength=len(k_range))[:len(k_range)]
    exp_full = pmf_theory * n_sup_runs
    mask = exp_full >= 5
    obs_filt = obs_full[mask].astype(float)
    exp_filt = exp_full[mask]
    if len(obs_filt) >= 3:
        scale = obs_filt.sum() / exp_filt.sum()
        exp_filt = exp_filt * scale
        chi2, p_chi2 = stats.chisquare(obs_filt, exp_filt)
        st.markdown(f"**卡方拟合优度检验**: chi^2={chi2:.2f}, p={p_chi2:.4f}")

    super_ia = np.diff(arrivals_super)
    if len(super_ia) >= 5:
        fig_sia, r2_s = plot_interarrival_histogram(
            super_ia, lam1 + lam2,
            title=f"叠加过程间隔分布 (理论 Exp(lambda={lam1+lam2}))"
        )
        if fig_sia is not None:
            st.plotly_chart(fig_sia, use_container_width=True)
        html_err, _ = format_error_html(np.mean(super_ia), 1.0 / (lam1 + lam2))
        st.markdown(f"叠加间隔均值: {html_err}", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="conclusion-box">
    <b>问题思考与结论</b><br>
    ① <b>叠加后为何仍是泊松过程？</b> 两个独立泊松过程的特征函数相乘仍为泊松过程的特征函数，参数相加。
    更直观地说，两个独立指数间隔的交错排序，等价于一个更高强度的指数间隔序列。<br>
    ② lambda_1={lam1}, lambda_2={lam2} 叠加后 lambda={lam1+lam2}，
    事件数期望从 lambda_1*T={lam1*T_sup:.1f} 和 lambda_2*T={lam2*T_sup:.1f}
    增至 (lambda_1+lambda_2)T={(lam1+lam2)*T_sup:.1f}。<br>
    ③ <b>应用：</b> 通信网络中多用户数据包到达、客服中心多线路来电均可建模为独立泊松过程的叠加。
    银行柜台的多队列合并、公路车流的汇入也是同样的数学原理。<br>
    ④ <b>动手实验：</b> 试试 λ₁=λ₂——叠加后强度翻倍，事件数分布的中心也翻倍。验证这个线性关系。
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 标签页 3：泊松过程稀释性
# ============================================================

def render_tab_thinning(lam, p, T_thin):
    st.markdown("## 泊松过程稀释性")

    st.markdown(f"""
    <div class="theory-box">
    <b>稀释定理</b>：设 N(t) 是强度为 lambda={lam} 的泊松过程。
    若每个事件以概率 p={p} 被<em>独立保留</em>，以概率 1-p={1-p:.2f} 被丢弃，
    则保留事件构成的计数过程 N_p(t) 仍为泊松过程，强度为 lambda*p = {lam*p:.2f}。<br>
    <b>物理意义</b>：对泊松事件流进行独立的随机筛选，筛选后的流仍为泊松流，强度按筛选比折减。
    好比每一辆经过收费站的汽车以一定概率被抽查——被抽查的车辆序列仍构成泊松过程（只是强度降低了）。<br>
    <b>Bernoulli 稀释</b>：每个事件独立地抛一枚"保留概率为 p"的硬币。这是最经典的随机筛选机制。
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight-box" style="font-size:0.85rem;">
    <b>如何阅读此图：</b> 上图为原始泊松过程（全部事件），下图为稀释后过程（绿色为保留事件，红色×为丢弃事件）。
    观察：① 稀释后的计数过程是否仍然平稳；② 保留事件数是否接近 λpT；③ 丢弃事件是否均匀散布（而非聚集）。<br>
    <b>核心验证：</b> 稀释后的间隔时间是否仍服从指数分布 Exp(λp)。
    </div>
    """, unsafe_allow_html=True)

    arrivals, _ = generate_poisson_process(lam, T_thin)

    if len(arrivals) > 0:
        keep_mask = np.random.random(len(arrivals)) < p
        retained = arrivals[keep_mask]
        discarded = arrivals[~keep_mask]
    else:
        retained = np.array([])
        discarded = np.array([])

    st.markdown("### 原始过程与稀释过程对比")

    fig_thin = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f'原始泊松过程 (lambda={lam})，总事件数={len(arrivals)}',
            f'稀释后过程 (p={p}, lambda*p={lam*p:.2f})，保留={len(retained)}，丢弃={len(discarded)}'
        )
    )

    ox, oy = make_step_xy(arrivals, T_thin)
    fig_thin.add_trace(go.Scatter(
        x=ox, y=oy, mode='lines', name='原始过程',
        line=dict(color='#3b82f6', width=2.2, shape='hv'),
        hovertemplate='t: %{x:.2f}<br>N(t): %{y}<extra></extra>'
    ), row=1, col=1)
    if len(arrivals) > 0:
        fig_thin.add_trace(go.Scatter(
            x=arrivals, y=list(range(1, len(arrivals) + 1)),
            mode='markers',
            marker=dict(size=5, color='#3b82f6', line=dict(width=0)),
            showlegend=False,
            hovertemplate='到达: %{x:.3f}<br>事件#: %{y}<extra></extra>'
        ), row=1, col=1)

    rx, ry = make_step_xy(retained, T_thin) if len(retained) > 0 else ([0, T_thin], [0, 0])
    fig_thin.add_trace(go.Scatter(
        x=rx, y=ry, mode='lines', name='稀释后过程 (仅保留)',
        line=dict(color='#22c55e', width=2.5, shape='hv'),
        hovertemplate='t: %{x:.2f}<br>N_p(t): %{y}<extra></extra>'
    ), row=2, col=1)

    if len(retained) > 0:
        fig_thin.add_trace(go.Scatter(
            x=retained, y=list(range(1, len(retained) + 1)),
            mode='markers',
            marker=dict(size=6, color='#22c55e', symbol='circle', line=dict(width=0)),
            name='保留事件',
            hovertemplate='保留: %{x:.3f}<br>事件#: %{y}<extra></extra>'
        ), row=2, col=1)

    if len(discarded) > 0:
        disc_y = np.searchsorted(retained, discarded) if len(retained) > 0 else np.zeros(len(discarded))
        fig_thin.add_trace(go.Scatter(
            x=discarded, y=disc_y,
            mode='markers',
            marker=dict(size=7, color='#ef4444', symbol='x', line=dict(width=1.5)),
            name='丢弃事件',
            hovertemplate='丢弃: %{x:.3f}<extra></extra>'
        ), row=2, col=1)

    fig_thin.update_xaxes(title_text='时间 t', row=2, col=1)
    fig_thin.update_yaxes(title_text='N(t)', row=1, col=1)
    fig_thin.update_yaxes(title_text='N_p(t)', row=2, col=1)
    fig_thin.update_layout(
        height=600, template='simple_white', hovermode='x unified',
        title=dict(text=f"泊松过程稀释 (lambda={lam}, p={p})", font=dict(size=16)),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    for i in range(1, 3):
        fig_thin.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9', row=i, col=1)
        fig_thin.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9', row=i, col=1)
    st.plotly_chart(fig_thin, use_container_width=True)

    st.markdown("### 理论值与模拟值对比")
    actual_ratio = len(retained) / len(arrivals) if len(arrivals) > 0 else 0
    results = [
        ("原始事件数", lam * T_thin, len(arrivals)),
        ("稀释后事件数", lam * p * T_thin, len(retained)),
        ("稀释比例", p, actual_ratio),
    ]
    show_error_table(results)

    # 多 p 值对比
    st.markdown("### 不同稀释概率 p 的效果对比")
    st.markdown("""
    <div class="highlight-box" style="font-size:0.85rem;">
    <b>实验设计：</b> 固定 λ 和 T，仅在 p=0.2, 0.4, 0.6, 0.8 下各模拟 200 次，取平均事件数。
    理论预测应为线性关系：平均事件数 = λpT（即与 p 成正比）。条形图越接近红色折线，稀释定理验证越成功。
    </div>
    """, unsafe_allow_html=True)
    p_values = [0.2, 0.4, 0.6, 0.8]
    n_multi = 200
    fig_multi = go.Figure()
    p_results = []

    for p_i in p_values:
        retained_counts = []
        for _ in range(n_multi):
            at_full, _ = generate_poisson_process(lam, T_thin)
            retained_counts.append(np.sum(np.random.random(len(at_full)) < p_i))
        avg_retained = np.mean(retained_counts)
        theory_val = lam * p_i * T_thin
        p_results.append((p_i, avg_retained, theory_val))

    p_vals = [r[0] for r in p_results]
    sim_vals = [r[1] for r in p_results]
    theory_vals = [r[2] for r in p_results]

    fig_multi.add_trace(go.Bar(
        x=p_vals, y=sim_vals, name='模拟值 (200次平均)',
        marker_color='#3b82f6', opacity=0.7,
        text=[f'{v:.2f}' for v in sim_vals], textposition='outside',
        hovertemplate='p: %{x}<br>模拟事件数: %{y:.2f}<extra></extra>'
    ))
    fig_multi.add_trace(go.Scatter(
        x=p_vals, y=theory_vals, mode='markers+lines',
        name='理论值 lambda*p*T',
        marker=dict(color='#ef4444', size=10), line=dict(color='#ef4444', width=2, dash='dash'),
        hovertemplate='p: %{x}<br>理论事件数: %{y:.1f}<extra></extra>'
    ))
    fig_multi.update_layout(
        title=dict(text=f"不同 p 值下稀释后事件数 (lambda={lam}, T={T_thin})", font=dict(size=16)),
        xaxis_title=dict(text='稀释概率 p', font=dict(size=14)),
        yaxis_title=dict(text='平均事件数', font=dict(size=14)),
        template='simple_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    st.plotly_chart(fig_multi, use_container_width=True)

    for p_i, sim_v, theory_v in p_results:
        _, rel = format_error_html(sim_v, theory_v)
        color_cls = get_error_color(rel)[1]
        st.markdown(
            f"p={p_i}：模拟 {sim_v:.2f} vs 理论 {theory_v:.2f}，"
            f"误差 <span class='{color_cls}'>{rel:.2%}</span>",
            unsafe_allow_html=True
        )

    st.markdown("### 稀释过程间隔分布验证")
    st.markdown("""
    <div class="highlight-box" style="font-size:0.85rem;">
    <b>核心检验：</b> 如果稀释后"仍为泊松过程"，则保留事件的间隔时间应服从 Exp(λp)。
    下方自动生成间隔直方图并与理论指数分布对比。同时进行 KS 检验——p >= 0.05 则验证成功。
    </div>
    """, unsafe_allow_html=True)
    if len(retained) >= 5:
        thin_ia = np.diff(retained)
        fig_tia, r2_t = plot_interarrival_histogram(
            thin_ia, lam * p,
            title=f"稀释后间隔分布 (理论 Exp(lambda*p={lam*p:.2f}))"
        )
        if fig_tia is not None:
            st.plotly_chart(fig_tia, use_container_width=True)
        html_err, _ = format_error_html(np.mean(thin_ia), 1.0 / (lam * p))
        st.markdown(f"稀释间隔均值: {html_err}", unsafe_allow_html=True)
        if r2_t is not None:
            st.markdown(f"**拟合优度 R^2** = {r2_t:.4f}")

        d_stat, p_val, ks_conc = ks_test_exponential(thin_ia, lam * p)
        if d_stat is not None:
            st.markdown(f"**KS 检验**: D={d_stat:.4f}, {ks_conc}", unsafe_allow_html=True)
    else:
        st.warning("保留事件数过少，请增大 p 或 lambda。")

    st.markdown(f"""
    <div class="conclusion-box">
    <b>问题思考与结论</b><br>
    ① <b>稀释后为何仍是泊松过程？</b> 每个事件独立以概率 p 保留，等价于对泊松过程作 Bernoulli 稀释。
    可证稀释后的计数过程满足泊松过程四条公理，强度为 lambda*p。增量分布为 Poisson(lambda*p*t)。
    独立筛选相当于给泊松过程的特征函数乘以常数因子——不会改变分布类型，只改变参数。<br>
    ② lambda={lam}, p={p} 时有效强度 lambda*p={lam*p:.2f}，
    事件数期望从 lambda*T={lam*T_thin:.1f} 降至 lambda*p*T={lam*p*T_thin:.1f}。<br>
    ③ <b>应用：</b> 网络数据包随机丢包模型、保险理赔中实际赔付的建模、客服中心骚扰电话过滤、
    放射性粒子探测中探测器效率校正等。<br>
    ④ <b>动手实验：</b> 试试 p=0.1（大量丢弃）和 p=0.9（几乎全保留），观察稀释后过程的稀疏程度变化。
    特别注意：p 的值并不等于原始事件数与保留事件数之比——那只是 p 的一个估计值。
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 标签页 4：客服中心实际应用（含 M/M/c 排队论）
# ============================================================

def render_tab_call_center(shift, n_lines, spam_ratio, n_agents, mu_service, T_shift_override):
    st.markdown("## 客服中心排队系统模拟")

    st.markdown("""
    <div class="theory-box">
    <b>应用背景</b>：客服中心电话接入符合泊松过程的原因-<br>
    ① 大量独立客户各自独立决定拨打；<br>
    ② 每个客户在短时间内的拨打概率很小（稀有事件）；<br>
    ③ 满足平稳独立增量假设。<br>
    利用泊松过程的<em>叠加性</em>（多线路合并）、<em>稀释性</em>（骚扰过滤）和<em>M/M/c 排队论</em>（Erlang C 公式）
    可有效建模和优化客服资源配置。
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight-box" style="font-size:0.85rem;">
    <b>本页面完整复现了一个客服中心的数学建模全流程：</b><br>
    <b>叠加</b> → 多条独立线路的来电合并为总进线流（泊松过程的叠加性）<br>
    <b>稀释</b> → 以一定概率过滤骚扰电话，得到有效来电（泊松过程的稀释性）<br>
    <b>排队</b> → 有效来电进入 M/M/c 排队系统，由 c 个坐席提供服务<br>
    <b>分析</b> → 使用 Erlang C 公式计算等待概率、平均等待时间等关键运营指标<br>
    每一步都清晰地展示了概率论如何从理论走向工程实践。
    </div>
    """, unsafe_allow_html=True)

    shift_map = {
        '早班(9-12点, lambda=2)':    (2.0, 3),
        '午班(12-18点, lambda=3.5)': (3.5, 6),
        '晚班(18-24点, lambda=1.5)': (1.5, 6),
    }
    lam_per_line, T_shift = shift_map[shift]

    if T_shift_override is not None:
        T_shift = T_shift_override

    line_arrivals = []
    for _ in range(n_lines):
        at, _ = generate_poisson_process(lam_per_line, T_shift)
        line_arrivals.append(at)

    all_arrivals = np.sort(np.concatenate(line_arrivals)) if line_arrivals else np.array([])

    if len(all_arrivals) > 0:
        valid_mask = np.random.random(len(all_arrivals)) >= spam_ratio
        valid_arrivals = all_arrivals[valid_mask]
        spam_arrivals = all_arrivals[~valid_mask]
    else:
        valid_arrivals = np.array([])
        spam_arrivals = np.array([])

    total_lam = n_lines * lam_per_line
    effective_lam = total_lam * (1 - spam_ratio)

    # ── 1. 总进线过程 ──
    st.markdown("### 第一步：多线路来电叠加（叠加性应用）")
    st.markdown(f"""
    <div class="highlight-box" style="font-size:0.85rem;">
    <b>叠加性实战：</b> {n_lines} 条独立线路，每条以强度 λ={lam_per_line}/h 产生来电，
    叠加后的总进线过程为强度 n·λ={total_lam}/h 的泊松过程。
    图中浅色细线为各线路的独立计数过程，红色粗线为总进线过程——观察总进线事件数是否接近 n·λ·T。<br>
    <b>为什么总进线仍是泊松过程？</b> 因为我们之前验证过：独立泊松过程的叠加仍是泊松过程，强度相加。
    </div>
    """, unsafe_allow_html=True)

    fig_cc1 = go.Figure()
    line_colors = ['#93c5fd', '#86efac', '#fcd34d', '#c4b5fd', '#67e8f9']

    for i, at in enumerate(line_arrivals):
        sx, sy = make_step_xy(at, T_shift)
        fig_cc1.add_trace(go.Scatter(
            x=sx, y=sy, mode='lines',
            name=f'线路 {i+1} (lambda={lam_per_line}, 来电={len(at)})',
            line=dict(color=line_colors[i % len(line_colors)], width=1.5, shape='hv'),
            hovertemplate='t: %{x:.2f}<br>线路{i+1}: %{y}<extra></extra>'
        ))

    sx_total, sy_total = make_step_xy(all_arrivals, T_shift)
    fig_cc1.add_trace(go.Scatter(
        x=sx_total, y=sy_total, mode='lines',
        name=f'总进线 (n*lambda={total_lam}, 总来电={len(all_arrivals)})',
        line=dict(color='#ef4444', width=3, shape='hv'),
        hovertemplate='t: %{x:.2f}<br>总进线: %{y}<extra></extra>'
    ))

    fig_cc1.update_layout(
        title=dict(text=f"客服中心多线路进线过程 ({shift.split('(')[0]})", font=dict(size=16)),
        xaxis_title=dict(text=f'时间 (小时，共{T_shift}h)', font=dict(size=14)),
        yaxis_title=dict(text='累计来电数', font=dict(size=14)),
        template='simple_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    fig_cc1.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
    fig_cc1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
    st.plotly_chart(fig_cc1, use_container_width=True)

    col_v1, col_v2, col_v3 = st.columns(3)
    col_v1.metric("理论总强度 n*lambda", f"{total_lam:.2f}/h")
    est_total_lam = len(all_arrivals) / T_shift
    col_v2.metric("模拟总强度", f"{est_total_lam:.2f}/h")
    err = abs(est_total_lam - total_lam) / total_lam if total_lam > 0 else 0
    col_v3.markdown(
        f"相对误差 <span class='{get_error_color(err)[1]}'>{err:.2%}</span>",
        unsafe_allow_html=True
    )

    # ── 2. 来电过滤 ──
    st.markdown("### 第二步：骚扰电话过滤（稀释性应用）")
    st.markdown(f"""
    <div class="highlight-box" style="font-size:0.85rem;">
    <b>稀释性实战：</b> 对总进线过程的每通来电，以概率 {spam_ratio:.0%} 标记为骚扰并丢弃（红色×），
    以概率 {1-spam_ratio:.0%} 保留为有效来电（绿色）。有效来电仍为泊松过程，强度 = 总强度·(1-骚扰比例) = {total_lam:.2f}·{1-spam_ratio:.2f} = {effective_lam:.2f}/h。<br>
    <b>为什么过滤后仍是泊松过程？</b> 我们在稀释性 Tab 中验证过：Bernoulli 随机筛选不改变泊松过程的本质，只改变强度。
    </div>
    """, unsafe_allow_html=True)

    fig_cc2 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f'原始总进线过程 ({len(all_arrivals)} 通来电)',
            f'过滤后有效来电 (骚扰比例={spam_ratio:.0%}, 有效={len(valid_arrivals)}, 骚扰={len(spam_arrivals)})'
        )
    )

    ox, oy = make_step_xy(all_arrivals, T_shift)
    fig_cc2.add_trace(go.Scatter(
        x=ox, y=oy, mode='lines', name='总来电',
        line=dict(color='#3b82f6', width=2.2, shape='hv'),
        hovertemplate='t: %{x:.2f}<br>总数: %{y}<extra></extra>'
    ), row=1, col=1)

    vx, vy = make_step_xy(valid_arrivals, T_shift)
    fig_cc2.add_trace(go.Scatter(
        x=vx, y=vy, mode='lines', name='有效来电',
        line=dict(color='#22c55e', width=2.5, shape='hv'),
        hovertemplate='t: %{x:.2f}<br>有效: %{y}<extra></extra>'
    ), row=2, col=1)

    if len(valid_arrivals) > 0:
        fig_cc2.add_trace(go.Scatter(
            x=valid_arrivals, y=list(range(1, len(valid_arrivals) + 1)),
            mode='markers',
            marker=dict(size=5, color='#22c55e', line=dict(width=0)),
            name='有效来电事件',
            hovertemplate='有效来电: %{x:.3f}<br>#%{y}<extra></extra>'
        ), row=2, col=1)

    if len(spam_arrivals) > 0:
        spam_y_vals = (
            np.searchsorted(valid_arrivals, spam_arrivals)
            if len(valid_arrivals) > 0
            else np.zeros(len(spam_arrivals))
        )
        fig_cc2.add_trace(go.Scatter(
            x=spam_arrivals, y=spam_y_vals,
            mode='markers',
            marker=dict(size=6, color='#ef4444', symbol='x', line=dict(width=1.2)),
            name='骚扰电话',
            hovertemplate='骚扰: %{x:.3f}<extra></extra>'
        ), row=2, col=1)

    fig_cc2.update_xaxes(title_text=f'时间 (小时，共{T_shift}h)', row=2, col=1)
    fig_cc2.update_yaxes(title_text='累计来电', row=1, col=1)
    fig_cc2.update_yaxes(title_text='累计有效来电', row=2, col=1)
    fig_cc2.update_layout(
        height=600, template='simple_white', hovermode='x unified',
        title=dict(text="骚扰电话过滤效果", font=dict(size=16)),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    for i in range(1, 3):
        fig_cc2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9', row=i, col=1)
        fig_cc2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9', row=i, col=1)
    st.plotly_chart(fig_cc2, use_container_width=True)

    # ── 3. M/M/c 排队模拟 ──
    st.markdown("### 第三步：M/M/c 排队系统模拟")
    st.markdown(f"""
    <div class="highlight-box" style="font-size:0.85rem;">
    <b>M/M/c 是什么？</b> 第一个 M 表示到达过程为泊松（Markovian），第二个 M 表示服务时间为指数分布（Markovian），
    c={n_agents} 是坐席数。这是排队论中最经典的多服务器模型。<br>
    <b>离散事件模拟原理：</b> 系统按时间顺序处理每个顾客的到达、开始服务和离开事件。
    每个顾客到达时，如果有空闲坐席则立即开始服务；否则进入 FIFO 队列等待。
    服务时间独立同分布于 Exp(μ={mu_service})。下方图表展示了队列长度和系统总人数的实时变化。<br>
    <b>如何阅读：</b> 红色虚线为坐席数 c。当系统人数超过 c 时，超出部分在排队等待。
    观察黄色区域（队列长度）的波动——高负载时段队列会堆积。
    </div>
    """, unsafe_allow_html=True)

    if len(valid_arrivals) > 0:
        result = simulate_mmc_queue(effective_lam, mu_service, n_agents, T_shift)
    else:
        result = {
            'arrivals': np.array([]), 'waits': np.array([]),
            'queue_len_over_time': ([0, T_shift], [0, 0]),
            'system_len_over_time': ([0, T_shift], [0, 0]),
            'n_total': 0, 'n_served': 0, 'avg_wait': 0, 'max_queue': 0,
        }

    fig_queue = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=('队列长度随时间变化', '系统中总人数随时间变化')
    )

    qt, qv = result['queue_len_over_time']
    fig_queue.add_trace(go.Scatter(
        x=qt, y=qv, mode='lines',
        name='队列长度 (等待中)',
        line=dict(color='#f59e0b', width=2),
        fill='tozeroy', fillcolor='rgba(245,158,11,0.1)',
        hovertemplate='t: %{x:.2f}<br>队列: %{y}<extra></extra>'
    ), row=1, col=1)

    st_t, sv = result['system_len_over_time']
    fig_queue.add_trace(go.Scatter(
        x=st_t, y=sv, mode='lines',
        name=f'系统中总人数 (服务中+等待)',
        line=dict(color='#3b82f6', width=2),
        fill='tozeroy', fillcolor='rgba(59,130,246,0.1)',
        hovertemplate='t: %{x:.2f}<br>总人数: %{y}<extra></extra>'
    ), row=2, col=1)
    fig_queue.add_hline(
        y=n_agents, line_dash="dash", line_color="#ef4444",
        annotation_text=f"坐席数={n_agents}", row=2, col=1
    )

    fig_queue.update_xaxes(title_text='时间 (小时)', row=2, col=1)
    fig_queue.update_yaxes(title_text='队列长度', row=1, col=1)
    fig_queue.update_yaxes(title_text='系统中人数', row=2, col=1)
    fig_queue.update_layout(
        height=550, template='simple_white', hovermode='x unified',
        title=dict(text=f"M/M/{n_agents} 排队系统状态 (lambda={effective_lam:.2f}, mu={mu_service})", font=dict(size=16)),
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig_queue, use_container_width=True)

    # 等待时间分布
    if len(result['waits']) > 0 and np.max(result['waits']) > 0:
        fig_waits = go.Figure()
        fig_waits.add_trace(go.Histogram(
            x=result['waits'] * 60,
            nbinsx=min(40, max(10, int(np.sqrt(len(result['waits']))))),
            name='等待时间',
            marker_color='#8b5cf6', opacity=0.7,
            hovertemplate='等待: %{x:.1f} 分钟<br>频数: %{y}<extra></extra>'
        ))
        fig_waits.add_vline(
            x=result['avg_wait'] * 60,
            line_dash="dash", line_color="#ef4444",
            annotation_text=f"平均: {result['avg_wait']*60:.1f} 分钟"
        )
        fig_waits.update_layout(
            title=dict(text="顾客等待时间分布", font=dict(size=16)),
            xaxis_title=dict(text='等待时间 (分钟)', font=dict(size=14)),
            yaxis_title=dict(text='频数', font=dict(size=14)),
            template='simple_white', hovermode='x unified',
            margin=dict(l=40, r=20, t=50, b=40)
        )
        fig_waits.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
        fig_waits.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
        st.plotly_chart(fig_waits, use_container_width=True)

    # ── 4. Erlang C + 运营指标仪表盘 ──
    st.markdown("### 第四步：运营指标与 Erlang C 分析")
    st.markdown("""
    <div class="highlight-box" style="font-size:0.85rem;">
    <b>Erlang C 公式</b> 由丹麦数学家 A.K. Erlang 于 1917 年推导，是排队论中最著名的公式之一。
    它给出了 M/M/c 系统中"一个新到达的顾客需要等待"的概率 P(W>0)，以及平均等待时间 Wq 和平均队列长度 Lq。
    这些指标是客服中心资源配置的核心依据。<br>
    <b>关键指标解读：</b><br>
    - <b>利用率 ρ</b>：坐席的繁忙程度。ρ = λ/(c·μ)。ρ 越接近 1 系统越拥挤，ρ >= 1 则队列无限增长（系统不稳定）。<br>
    - <b>等待概率 P(W>0)</b>：来电需要排队的概率。与坐席数 c 和利用率 ρ 密切相关。<br>
    - <b>平均等待时间 Wq</b>：顾客平均在队列中的等待时间。由 Little 定律 Lq = λ·Wq 与队列长度关联。<br>
    - <b>服务等级 SL</b>：在目标时间（如 1 分钟）内接通的来电比例。行业标准通常要求 SL >= 80%。
    </div>
    """, unsafe_allow_html=True)

    p_wait_ec, lq_ec, wq_ec, rho_ec = erlang_c(effective_lam, mu_service, n_agents)

    # 仪表盘
    col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
    with col_k1:
        delta = "目标 < 0.7" if rho_ec >= 0.85 else ("繁忙" if rho_ec >= 0.6 else "正常")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">坐席利用率 rho</div>
            <div class="metric-value">{rho_ec:.1%}</div>
            <div class="metric-delta">{delta}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k2:
        color_wait = "#16a34a" if p_wait_ec < 0.2 else ("#d97706" if p_wait_ec < 0.5 else "#dc2626")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">等待概率 P(wait>0)</div>
            <div class="metric-value" style="color:{color_wait}">{p_wait_ec:.1%}</div>
            <div class="metric-delta">Erlang C 公式</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k3:
        avg_wait_min = wq_ec * 60 if wq_ec != float('inf') else float('inf')
        w_str = f"{avg_wait_min:.1f} 分钟" if avg_wait_min != float('inf') else "inf"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">平均等待时间 Wq</div>
            <div class="metric-value">{w_str}</div>
            <div class="metric-delta">Erlang C 理论值</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k4:
        lq_str = f"{lq_ec:.2f} 人" if lq_ec != float('inf') else "inf"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">平均队列长度 Lq</div>
            <div class="metric-value">{lq_str}</div>
            <div class="metric-delta">利特尔定律 Lq=lambda*Wq</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k5:
        sim_wait = result['avg_wait'] * 60
        w_sim_str = f"{sim_wait:.1f} 分钟"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">模拟平均等待</div>
            <div class="metric-value">{w_sim_str}</div>
            <div class="metric-delta">离散事件模拟 ({result['n_total']} 顾客)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 模拟 vs Erlang C 对比")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("模拟到达数", f"{result['n_total']}", f"理论 lambda*T={effective_lam*T_shift:.1f}")
    r2.metric("模拟平均等待", f"{result['avg_wait']*60:.2f} 分钟",
              f"Erlang C: {avg_wait_min:.2f} 分钟" if avg_wait_min != float('inf') else "inf")
    r3.metric("最大队列长度", f"{result['max_queue']}")
    r4.metric("坐席配置", f"{n_agents} 人", f"rho={rho_ec:.1%}")

    if rho_ec >= 1.0:
        st.error(f"系统不稳定：rho={rho_ec:.2f} >= 1，到达率超过服务能力，队列将无限增长！请增加坐席数。")
    elif rho_ec >= 0.85:
        st.warning(f"系统高负载：rho={rho_ec:.2%}，顾客等待时间较长，建议考虑增加坐席。")
    elif rho_ec >= 0.6:
        st.info(f"系统负载适中：rho={rho_ec:.2%}，当前配置基本合理。")
    else:
        st.success(f"系统负载较低：rho={rho_ec:.2%}，坐席资源充足。")

    # 推荐坐席数
    st.markdown("#### 坐席数量推荐")
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        min_c = max(1, int(np.floor(effective_lam / mu_service)) + 1)
        recommendations = []
        for c in range(min_c, min_c + 8):
            pw, _, wq, rho_c = erlang_c(effective_lam, mu_service, c)
            if rho_c >= 1.0:
                continue
            sl = 1 - pw * exp(-(c * mu_service - effective_lam) * 1.0)
            recommendations.append((c, pw, wq * 60, rho_c, sl))
            if pw < 0.05:
                break

        st.markdown("| 坐席数 c | 利用率 rho | 等待概率 | 平均等待 | 服务等级 SL(1min) |")
        st.markdown("|:---:|:---:|:---:|:---:|:---:|")
        for c, pw, wq_m, rho_c, sl in recommendations:
            pw_str = f"<span style='color:{'#16a34a' if pw<0.1 else '#d97706' if pw<0.3 else '#dc2626'}'>{pw:.1%}</span>"
            wq_str = f"{wq_m:.1f}min" if wq_m != float('inf') else "inf"
            sl_str = f"<span style='color:{'#16a34a' if sl>0.8 else '#d97706' if sl>0.5 else '#dc2626'}'>{sl:.1%}</span>"
            st.markdown(f"| {c} | {rho_c:.1%} | {pw_str} | {wq_str} | {sl_str} |", unsafe_allow_html=True)

    with rec_col2:
        if recommendations:
            best = recommendations[-1]
            st.markdown(f"""
            <div class="conclusion-box">
            <b>推荐配置</b><br>
            当前有效到达率 lambda={effective_lam:.2f}/h，服务率 mu={mu_service}/h/坐席。<br><br>
            <b>推荐坐席数：{best[0]} 人</b><br>
            利用率：{best[3]:.1%}<br>
            等待概率：{best[1]:.1%}<br>
            平均等待：{best[2]:.1f} 分钟<br>
            1分钟内接通率：{best[4]:.1%}<br>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <div class="highlight-box">
        <b>服务等级 (Service Level)</b> 定义为在目标时间内接通的来电比例。
        行业标准通常要求 80% 的来电在 20 秒内接通。上表中 SL(1min) 表示 1分钟内的接通率。
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 稀释性验证")
    est_eff_lam = len(valid_arrivals) / T_shift
    cv1, cv2, cv3 = st.columns(3)
    cv1.metric("理论有效强度 lambda(1-s)", f"{effective_lam:.2f}/h")
    cv2.metric("模拟有效强度", f"{est_eff_lam:.2f}/h")
    err_eff = abs(est_eff_lam - effective_lam) / effective_lam if effective_lam > 0 else 0
    cv3.markdown(
        f"相对误差 <span class='{get_error_color(err_eff)[1]}'>{err_eff:.2%}</span>",
        unsafe_allow_html=True
    )

    st.markdown(f"""
    <div class="conclusion-box">
    <b>问题思考与结论</b><br>
    ① <b>叠加性应用：</b> {n_lines} 条独立线路的来电叠加为强度 {total_lam}/h 的总泊松过程，
    验证了独立泊松流合并定理。<br>
    ② <b>稀释性应用：</b> 以 {1-spam_ratio:.0%} 比例过滤骚扰电话后，
    有效来电仍为泊松过程（强度 {effective_lam:.2f}/h）。注意：过滤掉的骚扰电话不会进入排队系统。<br>
    ③ <b>排队论应用（核心）：</b> M/M/{n_agents} 排队系统的 Erlang C 公式给出了
    等待概率={p_wait_ec:.1%}、平均等待={avg_wait_min:.1f}min、利用率={rho_ec:.1%}。
    当 rho>=1 时系统不稳定，需增加坐席。<br>
    ④ <b>资源配置价值：</b> 泊松过程 + 排队论给出等待时间分布、溢出概率等关键指标，
    为客服中心科学排班提供数学基础。优化目标是在服务等级和人员成本之间取得平衡。<br>
    ⑤ <b>完整知识链：</b> 公理定义 → 基本性质 → 叠加/稀释 → 排队建模 → Erlang C → 资源优化。
    这就是概率论从数学定义到工程决策的完整路径。
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 标签页 5：总结
# ============================================================

def render_tab_summary():
    st.markdown("## 总结与展望")

    st.markdown("""
    <div class="hero-container" style="padding: 36px 48px;">
        <div style="font-size:1.5rem;font-weight:700;color:#93c5fd;position:relative;">从泊松过程到排队论</div>
        <div style="font-size:0.95rem;color:#94a3b8;margin-top:8px;position:relative;line-height:1.7;">
            我们完成了一次从数学定义出发、逐步深入实际应用的完整探索-
            泊松过程不仅是一个优美的数学模型，更是理解和优化现实世界中随机服务系统的核心工具。
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight-box" style="font-size:0.9rem;">
    <b>学习回顾：</b> 本工具按照"理解概念 → 验证性质 → 综合应用"的认知路径设计。
    如果某个 Tab 的内容还不太清晰，建议回到对应的 Tab 重新实验——
    每个 Tab 顶部的理论框提供了核心公式，各图表上方有"如何阅读"的引导说明，
    底部的结论框则有思考题帮助巩固理解。数学不是"看懂"的，而是"做懂"的。
    </div>
    """, unsafe_allow_html=True)

    col_1, col_2 = st.columns(2)

    with col_1:
        st.markdown("""
        <div class="theory-box">
        <b>基本性质</b><br>
        泊松过程 N(t) 是最基本的计数过程模型：<br>
        - N(t) ~ Poisson(lambda*t)<br>
        - 到达间隔 i.i.d. ~ Exp(lambda)<br>
        - <b>无记忆性</b>：未来与过去无关<br>
        - <b>平稳独立增量</b>：核心特征<br>
        <b>验证手段</b>：KS 检验、R^2 拟合优度、无记忆性可视化<br>
        <b>学习要点</b>：理解公理 → 性质的数学推导，而非死记硬背公式。
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="theory-box">
        <b>叠加性</b><br>
        独立泊松过程之和仍为泊松过程：<br>
        - lambda = lambda_1 + lambda_2<br>
        - 多源输入的合并模型<br>
        - 客服中心多条线路的总来电<br>
        <b>验证手段</b>：三图对比、卡方拟合优度、间隔分布验证<br>
        <b>学习要点</b>：代数封闭性——泊松过程经过叠加"仍是自己"。
        </div>
        """, unsafe_allow_html=True)

    with col_2:
        st.markdown("""
        <div class="theory-box">
        <b>稀释性</b><br>
        Bernoulli 随机筛选后仍为泊松过程：<br>
        - lambda' = lambda * p<br>
        - 每个事件独立保留/丢弃<br>
        - 骚扰电话过滤模型<br>
        <b>验证手段</b>：多 p 值对比、间隔分布验证、KS 检验<br>
        <b>学习要点</b>：Bernoulli 稀释不改变分布类型，只改变强度参数。
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="conclusion-box">
        <b>客服中心 M/M/c 排队</b><br>
        泊松过程性质在排队论中的综合应用：<br>
        - <b>叠加</b>：多线路来电合并<br>
        - <b>稀释</b>：骚扰电话过滤<br>
        - <b>Erlang C</b>：等待概率、平均队长<br>
        - <b>资源配置</b>：科学确定坐席数<br>
        <b>关键指标</b>：利用率 rho、等待概率 P(W>0)、服务等级 SL<br>
        <b>学习要点</b>：体验"性质 → 工程应用"的链路——从数学公理到运营决策的全过程。
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="highlight-box">
    <b>探索延伸</b><br>
    泊松过程的理论远不止于此，以下是值得进一步探索的方向：<br><br>
    - <b>非齐次泊松过程</b>：强度 lambda(t) 随时间变化，可用于建模峰谷时段来电模式<br>
    - <b>复合泊松过程</b>：每次跳跃大小随机，应用于保险精算中的总赔付额建模<br>
    - <b>M/G/c 排队系统</b>：服务时间服从一般分布（非指数），更接近实际场景<br>
    - <b>排队网络</b>：多个服务节点串联，如客服中心的 IVR -> 坐席 -> 主管升级流程<br>
    - <b>机器学习结合</b>：利用历史数据估计 lambda(t)，实现动态排班优化<br><br>
    <b>推荐学习路径：</b> 本工具 → 教科书理论推导 → 排队论专著（如 Gross & Harris）→ 运筹学教材 → 实际应用/论文。
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="conclusion-box">
    <b>核心收获</b><br>
    通过这次探索，我们深刻体会到：<br>
    ① 泊松过程的<b>公理化定义</b>看似简单（只有四条），但蕴含了丰富的数学性质——
    泊松分布、指数分布、无记忆性、叠加性、稀释性都可以从公理严格推导出来；<br>
    ② <b>叠加性</b>和<b>稀释性</b>使泊松过程在工程建模中异常灵活——合并和筛选后"仍是自己"，
    这种代数封闭性在随机过程中非常珍贵；<br>
    ③ 概率论不是空中楼阁，<b>Erlang C 公式</b>（1917 年推导）到今天仍在全球客服中心的路由系统中运行——
    数学的美在于经得起时间的考验；<br>
    ④ 交互式可视化让我们<b>直观感受</b>随机过程的演化——模拟不是替代理论，而是验证和深化理解理论的最好方式。
    你看到的每一条随机曲线，都是公理定义的必然结果。
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 主函数
# ============================================================

def main():
    st.title("泊松过程交互式可视化")
    st.markdown("《概率论与随机过程》期中作业  |  基本性质 · 叠加性 · 稀释性 · 客服中心 M/M/c 排队系统")

    # ── 侧边栏 ──
    with st.sidebar:
        st.header("控制面板")
        st.caption("调整下方参数，观察各 Tab 页中图表与统计量的实时变化。")

        st.markdown("---")
        st.markdown("### 基本性质参数")
        st.markdown("""
        <div class="theory-box" style="font-size:0.8rem; padding:12px 14px; margin-bottom:12px;">
        <b>lambda (λ)</b> — 泊松过程的<em>强度</em>，即单位时间内事件发生的平均次数。
        λ 越大，事件越密集。观察样本路径的跳跃频率如何随 λ 变化。<br>
        <b>时间范围 T</b> — 模拟窗口长度，控制了观察区间 [0, T]。T 越大，事件总数越多，统计规律越明显。<br>
        <b>样本量</b> — 重复生成样本路径的次数，增大可降低统计分布的随机波动，使直方图更接近理论 PMF/PDF。
        </div>
        """, unsafe_allow_html=True)
        lam_basic = st.selectbox(
            "强度 lambda", [0.5, 1.0, 2.0, 5.0], index=1,
            key="basic_lam"
        )
        T_basic = st.slider(
            "时间范围 T", 10.0, 100.0, 50.0, 5.0,
            key="basic_T"
        )
        n_basic = st.slider(
            "样本量", 100, 10000, 1000, 100,
            key="basic_n"
        )
        st.button("重新生成所有样本", key="basic_regenerate")

        st.markdown("---")
        st.markdown("### 叠加性参数")
        st.markdown("""
        <div class="theory-box" style="font-size:0.8rem; padding:12px 14px; margin-bottom:12px;">
        <b>lambda_1, lambda_2</b> — 两条独立泊松过程的强度。
        叠加后的总过程强度应为 λ₁ + λ₂。尝试让两者相差悬殊（如 0.5 + 4.0），观察叠加后谁"主导"了事件流。<br>
        <b>T</b> — 观察时间窗口，越长统计越稳定。
        </div>
        """, unsafe_allow_html=True)
        lam1_sup = st.slider("lambda_1", 0.1, 5.0, 1.0, 0.1, key="sup_lam1")
        lam2_sup = st.slider("lambda_2", 0.1, 5.0, 1.5, 0.1, key="sup_lam2")
        T_sup = st.slider("时间范围 T", 10.0, 100.0, 50.0, 5.0, key="sup_T")
        st.button("生成叠加过程", key="sup_generate")

        st.markdown("---")
        st.markdown("### 稀释性参数")
        st.markdown("""
        <div class="theory-box" style="font-size:0.8rem; padding:12px 14px; margin-bottom:12px;">
        <b>原始 lambda</b> — 稀释前泊松过程的强度。<br>
        <b>稀释概率 p</b> — 每个事件被<em>保留</em>的概率。稀释后有效强度 = λ · p。
        试试极端值：p → 1（几乎全部保留，与原始无明显差异）或 p → 0（大量丢弃，事件稀疏）。<br>
        <b>T</b> — 观察时间窗口。
        </div>
        """, unsafe_allow_html=True)
        lam_thin = st.slider("原始 lambda", 0.1, 5.0, 2.0, 0.1, key="thin_lam")
        p_thin = st.slider("稀释概率 p", 0.0, 1.0, 0.6, 0.05, key="thin_p")
        T_thin = st.slider("时间范围 T", 10.0, 100.0, 50.0, 5.0, key="thin_T")
        st.button("生成稀释过程", key="thin_generate")

        st.markdown("---")
        st.markdown("### 客服中心参数")
        st.markdown("""
        <div class="theory-box" style="font-size:0.8rem; padding:12px 14px; margin-bottom:12px;">
        <b>时段选择</b> — 不同时段的单线来电强度 λ 不同。午班最忙（λ=3.5），晚班最闲（λ=1.5），反映实际呼叫中心的峰谷规律。<br>
        <b>客服线路数</b> — 独立来电线路的数量，模拟多条线路合并（叠加性）。线路越多，总进线强度越大。<br>
        <b>骚扰电话比例</b> — 来电中被过滤掉的骚扰比例，应用稀释性。比例越高，有效来电越少。
        </div>
        """, unsafe_allow_html=True)
        shift_cc = st.selectbox(
            "时段选择",
            ['早班(9-12点, lambda=2)', '午班(12-18点, lambda=3.5)', '晚班(18-24点, lambda=1.5)'],
            index=1, key="cc_shift"
        )
        lines_cc = st.slider("客服线路数", 1, 5, 3, 1, key="cc_lines")
        spam_cc = st.slider("骚扰电话比例", 0.0, 0.5, 0.2, 0.05, key="cc_spam")

        st.markdown("---")
        st.markdown("**M/M/c 排队参数**")
        st.markdown("""
        <div class="theory-box" style="font-size:0.8rem; padding:12px 14px; margin-bottom:12px;">
        <b>坐席数 c</b> — M/M/c 中的 c，即同时服务的坐席数量。c 越大排队时间越短，但成本越高。核心优化变量。<br>
        <b>服务率 μ</b> — 每个坐席每小时能处理的来电数。μ=6 即平均每通电话处理 10 分钟（1/6 小时）。<br>
        <b>模拟时长</b> — 离散事件模拟的时钟长度。设为 0 使用时段默认时长。
        </div>
        """, unsafe_allow_html=True)
        n_agents_cc = st.slider("坐席数 c", 1, 20, 4, 1, key="cc_agents")
        mu_service_cc = st.slider(
            "服务率 mu (/h/坐席)", 1.0, 15.0, 6.0, 0.5,
            key="cc_mu",
            help="每个坐席每小时可处理的来电数。mu=6 即平均每通电话处理 10 分钟。"
        )
        T_shift_override_cc = st.slider(
            "模拟时长 (h)", 1.0, 12.0, 0.0, 0.5,
            key="cc_T_override",
            help="设为 0 则使用时段默认时长。"
        )
        T_shift_override_cc = T_shift_override_cc if T_shift_override_cc > 0 else None

        st.button("开始模拟", key="cc_simulate")

    # ── 标签页 ──
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "首页总览",
        "基本性质",
        "叠加性",
        "稀释性",
        "客服中心应用",
        "总结与展望"
    ])

    with tab0:
        render_tab_home()

    with tab1:
        render_tab_basic(lam_basic, T_basic, n_basic)

    with tab2:
        render_tab_superposition(lam1_sup, lam2_sup, T_sup)

    with tab3:
        render_tab_thinning(lam_thin, p_thin, T_thin)

    with tab4:
        render_tab_call_center(shift_cc, lines_cc, spam_cc, n_agents_cc, mu_service_cc, T_shift_override_cc)

    with tab5:
        render_tab_summary()


if __name__ == "__main__":
    main()
