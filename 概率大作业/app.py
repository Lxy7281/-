# -*- coding: utf-8 -*-
"""
泊松过程交互式可视化应用
=========================
满足《概率论与随机过程》期中作业全部要求：
  - 基本性质、叠加性、稀释性、客服中心应用
  - 理论值与模拟值对比，学术严谨+交互友好
  - 支持通过 Streamlit Print 导出为独立 HTML 文件

运行方法：
  pip install streamlit numpy plotly scipy pandas
  streamlit run app.py

导出 HTML：
  点击浏览器右上角 ⋮ → Print → 目标选择 "另存为 HTML"
"""

import sys
import os

# Windows 终端 UTF-8 编码修复
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 页面配置 ─────────────────────────────────────────────
st.set_page_config(page_title="泊松过程交互式可视化", layout="wide")

# ── 自定义 CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .theory-box {
        background: linear-gradient(135deg, #e8f0fe 0%, #f3e8ff 100%);
        border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 12px 0;
    }
    .conclusion-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border-left: 4px solid #22c55e;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 12px 0;
    }
    .metric-green  { color: #16a34a; font-weight: bold; }
    .metric-yellow { color: #d97706; font-weight: bold; }
    .metric-red    { color: #dc2626; font-weight: bold; }
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
    """将到达时刻转换为阶梯图的 (x, y) 坐标，用于 line_shape='hv'"""
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
    """计算观测值与期望值的决定系数 R²"""
    ss_res = np.sum((observed - expected) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1 - ss_res / ss_tot


def format_error_html(sim_val, theory_val):
    """生成带颜色标注的误差 HTML，返回 (html_str, rel_err)"""
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
    """多次生成泊松过程并汇总所有间隔时间，用于统计分析"""
    all_interarrivals = []
    for _ in range(n_runs):
        _, ia = generate_poisson_process(lam, T)
        all_interarrivals.extend(ia.tolist())
    return np.array(all_interarrivals)


def plot_interarrival_histogram(interarrivals, lam, title="到达间隔时间分布"):
    """绘制间隔时间直方图 + 理论指数分布 PDF，返回 (fig, R²)"""
    if len(interarrivals) < 5:
        return None, None

    nbins = min(50, max(10, int(np.sqrt(len(interarrivals)))))
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=interarrivals, histnorm='probability density',
        nbinsx=nbins,
        name='模拟数据', marker_color='#2196F3', opacity=0.65,
        hovertemplate='间隔: %{x:.3f}<br>密度: %{y:.4f}<extra></extra>'
    ))

    x_max = np.max(interarrivals) * 1.1
    x_theory = np.linspace(0, x_max, 200)
    pdf_theory = lam * np.exp(-lam * x_theory)
    fig.add_trace(go.Scatter(
        x=x_theory, y=pdf_theory, mode='lines',
        name=f'理论 Exp(λ={lam})',
        line=dict(color='#F44336', width=2.5, dash='dash'),
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
        template='plotly_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    return fig, r2


def show_error_table(results):
    """显示理论值 vs 模拟值对比表格（带颜色标注）"""
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


# ============================================================
# 标签页 1：泊松过程基本性质
# ============================================================

def render_tab_basic(lam, T, n_samples):
    st.markdown("## 泊松过程基本性质")

    # 理论说明
    st.markdown(f"""
    <div class="theory-box">
    <b>定义</b>：计数过程 $\\{{N(t), t \\geq 0\\}}$ 称为强度为 $\\lambda$ 的<em>泊松过程</em>，若满足：<br>
    &emsp;① $N(0)=0$；&emsp;② 独立增量；&emsp;③ 平稳增量；
    &emsp;④ $P(N(h)=1)=\\lambda h+o(h)$，$P(N(h)\\geq 2)=o(h)$。<br><br>
    <b>物理意义</b>：$\\lambda={lam}$ 表示单位时间内事件发生的平均次数（强度）。
    间隔时间独立同分布于指数分布 $\\text{{Exp}}(\\lambda={lam})$。
    期望事件数 $E[N(t)]=\\lambda t={lam}t$，方差 $\\text{{Var}}[N(t)]=\\lambda t$。
    </div>
    """)

    # ── 区域 1：样本路径 ──
    st.markdown("### 样本路径可视化")
    colors = ['#2196F3', '#E91E63', '#4CAF50']

    arrivals_list = []
    for _ in range(3):
        at, _ = generate_poisson_process(lam, T)
        arrivals_list.append(at)

    fig_paths = go.Figure()
    for i, at in enumerate(arrivals_list):
        sx, sy = make_step_xy(at, T)
        fig_paths.add_trace(go.Scatter(
            x=sx, y=sy, mode='lines',
            name=f'路径 {i+1} (N(T)={len(at)}, λ̂={len(at)/T:.2f})',
            line=dict(color=colors[i], width=2.2, shape='hv'),
            hovertemplate='t: %{{x:.2f}}<br>N(t): %{{y}}<extra></extra>'
        ))
        if len(at) > 0:
            fig_paths.add_trace(go.Scatter(
                x=at, y=list(range(1, len(at) + 1)),
                mode='markers',
                marker=dict(size=5, color=colors[i], line=dict(width=0)),
                showlegend=False,
                hovertemplate='到达时间: %{{x:.3f}}<br>事件编号: %{{y}}<extra></extra>'
            ))

    fig_paths.update_layout(
        title=dict(text=f"泊松过程样本路径 (λ={lam})", font=dict(size=16)),
        xaxis_title=dict(text='时间 t', font=dict(size=14)),
        yaxis_title=dict(text='N(t)', font=dict(size=14)),
        template='plotly_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    fig_paths.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig_paths.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    st.plotly_chart(fig_paths, use_container_width=True)

    # ── 区域 2：间隔时间分析 ──
    st.markdown("### 到达间隔时间分析")
    n_runs = max(1, n_samples // 10)
    all_ia = collect_interarrivals(lam, T, n_runs)
    fig_hist, r2 = plot_interarrival_histogram(all_ia, lam,
        title=f"到达间隔时间分布 (λ={lam}, 样本数={len(all_ia)})")
    if fig_hist is not None:
        st.plotly_chart(fig_hist, use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    sim_mean = np.mean(all_ia)
    theory_mean = 1.0 / lam
    html_err, rel_err = format_error_html(sim_mean, theory_mean)
    col_a.metric("模拟均值", f"{sim_mean:.4f}", f"理论 {theory_mean:.4f}")
    col_b.metric("理论均值 1/λ", f"{theory_mean:.4f}")
    col_c.markdown(
        f"**相对误差** <span class='{get_error_color(rel_err)[1]}'>{rel_err:.2%}</span>",
        unsafe_allow_html=True
    )
    if r2 is not None:
        r2_color = get_error_color(max(0, 1 - r2))[0] if r2 < 0.99 else '#16a34a'
        st.markdown(
            f"**拟合优度 R²** = <span style='color:{r2_color};font-weight:bold'>{r2:.4f}</span>",
            unsafe_allow_html=True
        )

    # ── 区域 3：无记忆性验证 ──
    st.markdown("### 无记忆性验证")
    st.markdown(
        "指数分布的无记忆性：$P(X > s+t \\mid X > t) = P(X > s)$，"
        "即给定已等待 $t_0$ 时间，剩余等待时间仍服从 $\\text{Exp}(\\lambda)$。"
    )

    col_t0, col_btn, _ = st.columns([1, 1, 3])
    with col_t0:
        t0 = st.number_input(
            "条件时间 t₀", value=5.0, min_value=0.1, max_value=50.0, step=0.5,
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
        marker_color='#2196F3', opacity=0.55,
        hovertemplate='间隔: %{x:.3f}<br>密度: %{y:.4f}<extra></extra>'
    ))
    fig_memory.add_trace(go.Histogram(
        x=remaining, histnorm='probability density', nbinsx=bins_mem,
        name=f't₀={t0} 后剩余时间 (n={len(remaining)})',
        marker_color='#FF9800', opacity=0.55,
        hovertemplate='剩余时间: %{x:.3f}<br>密度: %{y:.4f}<extra></extra>'
    ))
    x_max = max(np.max(X), np.max(remaining) if len(remaining) > 0 else 0) * 1.1
    x_theory = np.linspace(0, max(x_max, 0.1), 300)
    pdf_theory = lam * np.exp(-lam * x_theory)
    fig_memory.add_trace(go.Scatter(
        x=x_theory, y=pdf_theory, mode='lines',
        name=f'理论 Exp(λ={lam})',
        line=dict(color='#F44336', width=2.5, dash='dash'),
        hovertemplate='x: %{x:.3f}<br>f(x): %{y:.4f}<extra></extra>'
    ))

    fig_memory.update_layout(
        title=dict(text=f"无记忆性验证 (λ={lam}, t₀={t0})", font=dict(size=16)),
        xaxis_title=dict(text='间隔时间', font=dict(size=14)),
        yaxis_title=dict(text='概率密度', font=dict(size=14)),
        template='plotly_white', hovermode='x unified',
        barmode='overlay',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    fig_memory.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig_memory.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    st.plotly_chart(fig_memory, use_container_width=True)

    c1, c2 = st.columns(2)
    mean_all = np.mean(X)
    mean_rem = np.mean(remaining) if len(remaining) > 0 else 0
    c1.metric("全部间隔时间均值", f"{mean_all:.4f}", f"理论 1/λ={theory_mean:.4f}")
    c2.metric("t₀ 后剩余时间均值", f"{mean_rem:.4f}", f"理论 1/λ={theory_mean:.4f}")
    diff = abs(mean_all - mean_rem)
    st.markdown(f"**两均值之差的绝对值** = {diff:.4f}（越小越验证无记忆性）")

    # ── 问题思考与结论 ──
    st.markdown("""
    <div class="conclusion-box">
    <b>📝 问题思考与结论</b><br>
    ① <b>到达间隔为何服从指数分布？</b> 由泊松过程平稳独立增量性质可推导间隔时间 $T_i$ 满足
    $P(T_1 > t) = P(N(t)=0) = e^{-\\lambda t}$，故 $T_i \\sim \\text{Exp}(\\lambda)$。<br>
    ② <b>无记忆性的实际意义：</b> 无论系统已运行多久，下一事件的等待时间分布始终不变，这简化了排队系统的分析。<br>
    ③ <b>样本路径特征：</b> 观察阶梯图可知 $N(t)$ 单调不减、跃度恒为 1（几乎处处），符合泊松过程定义。
    </div>
    """)


# ============================================================
# 标签页 2：泊松过程叠加性
# ============================================================

def render_tab_superposition(lam1, lam2, T_sup):
    st.markdown("## 泊松过程叠加性")

    st.markdown(f"""
    <div class="theory-box">
    <b>叠加定理</b>：设 $\\{{N_1(t)\\}}$ 和 $\\{{N_2(t)\\}}$ 为两个<em>独立</em>的泊松过程，
    强度分别为 $\\lambda_1={lam1}, \\lambda_2={lam2}$，
    则叠加过程 $\\{{N(t) = N_1(t) + N_2(t)\\}}$ 仍为泊松过程，强度为
    $\\lambda = \\lambda_1 + \\lambda_2 = {lam1+lam2}$。<br>
    <b>物理意义</b>：多条独立泊松事件流的合并仍为泊松流，强度为各分流强度之和。
    这是排队论中多源输入合并的理论基础。
    </div>
    """)

    # 生成三个过程的样本路径
    arrivals1, _ = generate_poisson_process(lam1, T_sup)
    arrivals2, _ = generate_poisson_process(lam2, T_sup)
    arrivals_super = np.sort(np.concatenate([arrivals1, arrivals2]))

    # ── 三图对比 ──
    st.markdown("### 过程对比")

    fig_super = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f'过程 1 (λ₁={lam1})，事件数={len(arrivals1)}',
            f'过程 2 (λ₂={lam2})，事件数={len(arrivals2)}',
            f'叠加过程 (λ₁+λ₂={lam1+lam2})，事件数={len(arrivals_super)}'
        )
    )

    data_sets = [
        (arrivals1, '#2196F3', '过程 1'),
        (arrivals2, '#E91E63', '过程 2'),
        (arrivals_super, '#4CAF50', '叠加过程'),
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
    fig_super.update_yaxes(title_text='N₁(t)', row=1, col=1)
    fig_super.update_yaxes(title_text='N₂(t)', row=2, col=1)
    fig_super.update_yaxes(title_text='N(t)', row=3, col=1)
    fig_super.update_layout(
        height=650, template='plotly_white', hovermode='x unified',
        title=dict(text="独立泊松过程与其叠加过程对比", font=dict(size=16)),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    for i in range(1, 4):
        fig_super.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', row=i, col=1)
        fig_super.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', row=i, col=1)
    st.plotly_chart(fig_super, use_container_width=True)

    # ── 结果验证表格 ──
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

    # ── 间隔分布验证 ──
    st.markdown("### 叠加过程间隔分布验证")
    if st.button("分析叠加过程间隔分布", key="sup_analyze_btn"):
        super_ia = np.diff(arrivals_super)
        if len(super_ia) >= 5:
            fig_sia, r2_s = plot_interarrival_histogram(
                super_ia, lam1 + lam2,
                title=f"叠加过程间隔分布 (理论 Exp(λ={lam1+lam2}))"
            )
            if fig_sia is not None:
                st.plotly_chart(fig_sia, use_container_width=True)
            html_err, _ = format_error_html(np.mean(super_ia), 1.0 / (lam1 + lam2))
            st.markdown(f"叠加间隔均值: {html_err}", unsafe_allow_html=True)
            if r2_s is not None:
                st.markdown(f"**拟合优度 R²** = {r2_s:.4f}")
        else:
            st.warning("叠加事件数过少，请增大 λ 或时间范围。")

    # ── 思考与结论 ──
    st.markdown(f"""
    <div class="conclusion-box">
    <b>📝 问题思考与结论</b><br>
    ① <b>叠加后为何仍是泊松过程？</b> 两个独立泊松过程的特征函数相乘仍为泊松过程的特征函数，参数相加。
    直观理解：独立稀有事件流的合并仍为稀有事件流，强度自然叠加。<br>
    ② <b>λ₁={lam1}, λ₂={lam2}</b> 叠加后 <b>λ={lam1+lam2}</b>，
    事件数期望从 λ₁T={lam1*T_sup:.1f} 和 λ₂T={lam2*T_sup:.1f}
    增至 (λ₁+λ₂)T={(lam1+lam2)*T_sup:.1f}。<br>
    ③ <b>应用：</b> 通信网络中多用户数据包到达、客服中心多线路来电均可建模为独立泊松过程的叠加。
    </div>
    """)


# ============================================================
# 标签页 3：泊松过程稀释性
# ============================================================

def render_tab_thinning(lam, p, T_thin):
    st.markdown("## 泊松过程稀释性")

    st.markdown(f"""
    <div class="theory-box">
    <b>稀释定理</b>：设 $\\{{N(t)\\}}$ 是强度为 $\\lambda={lam}$ 的泊松过程。
    若每个事件以概率 $p={p}$ 被<em>独立保留</em>，以概率 $1-p={1-p:.2f}$ 被丢弃，
    则保留事件构成的计数过程 $\\{{N_p(t)\\}}$ 仍为泊松过程，强度为 $\\lambda p = {lam*p:.2f}$。<br>
    <b>物理意义</b>：对泊松事件流进行独立的随机筛选，筛选后的流仍为泊松流，强度按筛选比折减。
    </div>
    """)

    arrivals, _ = generate_poisson_process(lam, T_thin)

    if len(arrivals) > 0:
        keep_mask = np.random.random(len(arrivals)) < p
        retained = arrivals[keep_mask]
        discarded = arrivals[~keep_mask]
    else:
        retained = np.array([])
        discarded = np.array([])

    # ── 两图对比 ──
    st.markdown("### 原始过程与稀释过程对比")

    fig_thin = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f'原始泊松过程 (λ={lam})，总事件数={len(arrivals)}',
            f'稀释后过程 (p={p}, λp={lam*p:.2f})，保留={len(retained)}，丢弃={len(discarded)}'
        )
    )

    # 上图：原始过程
    ox, oy = make_step_xy(arrivals, T_thin)
    fig_thin.add_trace(go.Scatter(
        x=ox, y=oy, mode='lines', name='原始过程',
        line=dict(color='#2196F3', width=2.2, shape='hv'),
        hovertemplate='t: %{x:.2f}<br>N(t): %{y}<extra></extra>'
    ), row=1, col=1)
    if len(arrivals) > 0:
        fig_thin.add_trace(go.Scatter(
            x=arrivals, y=list(range(1, len(arrivals) + 1)),
            mode='markers',
            marker=dict(size=5, color='#2196F3', line=dict(width=0)),
            showlegend=False,
            hovertemplate='到达: %{x:.3f}<br>事件#: %{y}<extra></extra>'
        ), row=1, col=1)

    # 下图：稀释后（保留 + 丢弃用不同颜色标记）
    rx, ry = make_step_xy(retained, T_thin) if len(retained) > 0 else ([0, T_thin], [0, 0])
    fig_thin.add_trace(go.Scatter(
        x=rx, y=ry, mode='lines', name='稀释后过程 (仅保留)',
        line=dict(color='#4CAF50', width=2.5, shape='hv'),
        hovertemplate='t: %{x:.2f}<br>N_p(t): %{y}<extra></extra>'
    ), row=2, col=1)

    if len(retained) > 0:
        fig_thin.add_trace(go.Scatter(
            x=retained, y=list(range(1, len(retained) + 1)),
            mode='markers',
            marker=dict(size=6, color='#4CAF50', symbol='circle', line=dict(width=0)),
            name='保留事件',
            hovertemplate='保留: %{x:.3f}<br>事件#: %{y}<extra></extra>'
        ), row=2, col=1)

    if len(discarded) > 0:
        disc_y = np.searchsorted(retained, discarded) if len(retained) > 0 else np.zeros(len(discarded))
        fig_thin.add_trace(go.Scatter(
            x=discarded, y=disc_y,
            mode='markers',
            marker=dict(size=7, color='#F44336', symbol='x', line=dict(width=1.5)),
            name='丢弃事件',
            hovertemplate='丢弃: %{x:.3f}<extra></extra>'
        ), row=2, col=1)

    fig_thin.update_xaxes(title_text='时间 t', row=2, col=1)
    fig_thin.update_yaxes(title_text='N(t)', row=1, col=1)
    fig_thin.update_yaxes(title_text='N_p(t)', row=2, col=1)
    fig_thin.update_layout(
        height=600, template='plotly_white', hovermode='x unified',
        title=dict(text=f"泊松过程稀释 (λ={lam}, p={p})", font=dict(size=16)),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    for i in range(1, 3):
        fig_thin.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', row=i, col=1)
        fig_thin.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', row=i, col=1)
    st.plotly_chart(fig_thin, use_container_width=True)

    # ── 结果验证表格 ──
    st.markdown("### 理论值与模拟值对比")

    actual_ratio = len(retained) / len(arrivals) if len(arrivals) > 0 else 0
    results = [
        ("原始事件数", lam * T_thin, len(arrivals)),
        ("稀释后事件数", lam * p * T_thin, len(retained)),
        ("稀释比例", p, actual_ratio),
    ]
    show_error_table(results)

    # ── 间隔分布验证 ──
    st.markdown("### 稀释过程间隔分布验证")
    if st.button("分析稀释过程间隔分布", key="thin_analyze_btn"):
        if len(retained) >= 5:
            thin_ia = np.diff(retained)
            fig_tia, r2_t = plot_interarrival_histogram(
                thin_ia, lam * p,
                title=f"稀释后间隔分布 (理论 Exp(λp={lam*p:.2f}))"
            )
            if fig_tia is not None:
                st.plotly_chart(fig_tia, use_container_width=True)
            html_err, _ = format_error_html(np.mean(thin_ia), 1.0 / (lam * p))
            st.markdown(f"稀释间隔均值: {html_err}", unsafe_allow_html=True)
            if r2_t is not None:
                st.markdown(f"**拟合优度 R²** = {r2_t:.4f}")
        else:
            st.warning("保留事件数过少，请增大 p 或 λ。")

    # ── 思考与结论 ──
    st.markdown(f"""
    <div class="conclusion-box">
    <b>📝 问题思考与结论</b><br>
    ① <b>稀释后为何仍是泊松过程？</b> 每个事件独立以概率 p 保留，等价于对泊松过程作 Bernoulli 稀释。
    可证稀释后的计数过程满足泊松过程四条公理，强度为 λp。增量分布为 Poisson(λpt)。<br>
    ② <b>λ={lam}, p={p}</b> 时有效强度 <b>λp={lam*p:.2f}</b>，
    事件数期望从 λT={lam*T_thin:.1f} 降至 λpT={lam*p*T_thin:.1f}。<br>
    ③ <b>应用：</b> 网络数据包随机丢包模型、保险理赔中实际赔付的建模、客服中心骚扰电话过滤等。
    </div>
    """)


# ============================================================
# 标签页 4：客服中心实际应用
# ============================================================

def render_tab_call_center(shift, n_lines, spam_ratio):
    st.markdown("## 客服中心实际应用")

    st.markdown("""
    <div class="theory-box">
    <b>应用背景</b>：客服中心电话接入符合泊松过程的原因——<br>
    &emsp;① 大量独立客户各自独立决定拨打；<br>
    &emsp;② 每个客户在短时间内的拨打概率很小（稀有事件）；<br>
    &emsp;③ 满足平稳独立增量假设。<br>
    利用泊松过程的<em>叠加性</em>（多线路合并）和<em>稀释性</em>（骚扰过滤）可有效建模和优化客服资源配置。
    </div>
    """)

    shift_map = {
        '早班(9-12点, λ=2)':    (2.0, 3),
        '午班(12-18点, λ=3.5)': (3.5, 6),
        '晚班(18-24点, λ=1.5)': (1.5, 6),
    }
    lam_per_line, T_shift = shift_map[shift]

    # 生成各线路的独立泊松过程
    line_arrivals = []
    for _ in range(n_lines):
        at, _ = generate_poisson_process(lam_per_line, T_shift)
        line_arrivals.append(at)

    # 叠加：总来电
    all_arrivals = np.sort(np.concatenate(line_arrivals)) if line_arrivals else np.array([])

    # 稀释：过滤骚扰电话
    if len(all_arrivals) > 0:
        valid_mask = np.random.random(len(all_arrivals)) >= spam_ratio
        valid_arrivals = all_arrivals[valid_mask]
        spam_arrivals = all_arrivals[~valid_mask]
    else:
        valid_arrivals = np.array([])
        spam_arrivals = np.array([])

    total_lam = n_lines * lam_per_line
    effective_lam = total_lam * (1 - spam_ratio)

    # ── 总进线过程 ──
    st.markdown("### 总进线过程模拟")

    fig_cc1 = go.Figure()
    line_colors = ['#90CAF9', '#A5D6A7', '#FFCC80', '#CE93D8', '#80CBC4']

    for i, at in enumerate(line_arrivals):
        sx, sy = make_step_xy(at, T_shift)
        fig_cc1.add_trace(go.Scatter(
            x=sx, y=sy, mode='lines',
            name=f'线路 {i+1} (λ={lam_per_line}, 来电={len(at)})',
            line=dict(color=line_colors[i % len(line_colors)], width=1.5, shape='hv'),
            hovertemplate='t: %{x:.2f}<br>线路{i+1}: %{y}<extra></extra>'
        ))

    sx_total, sy_total = make_step_xy(all_arrivals, T_shift)
    fig_cc1.add_trace(go.Scatter(
        x=sx_total, y=sy_total, mode='lines',
        name=f'总进线 (nλ={total_lam}, 总来电={len(all_arrivals)})',
        line=dict(color='#F44336', width=3, shape='hv'),
        hovertemplate='t: %{x:.2f}<br>总进线: %{y}<extra></extra>'
    ))

    fig_cc1.update_layout(
        title=dict(text=f"客服中心多线路进线过程 ({shift.split('(')[0]})", font=dict(size=16)),
        xaxis_title=dict(text=f'时间 (小时，共{T_shift}h)', font=dict(size=14)),
        yaxis_title=dict(text='累计来电数', font=dict(size=14)),
        template='plotly_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    fig_cc1.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig_cc1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    st.plotly_chart(fig_cc1, use_container_width=True)

    # 验证叠加性
    est_total_lam = len(all_arrivals) / T_shift
    col_v1, col_v2, col_v3 = st.columns(3)
    col_v1.metric("理论总强度 nλ", f"{total_lam:.2f}/h")
    col_v2.metric("模拟总强度", f"{est_total_lam:.2f}/h")
    err = abs(est_total_lam - total_lam) / total_lam if total_lam > 0 else 0
    col_v3.markdown(
        f"相对误差 <span class='{get_error_color(err)[1]}'>{err:.2%}</span>",
        unsafe_allow_html=True
    )

    # ── 来电过滤效果 ──
    st.markdown("### 来电过滤效果（骚扰电话过滤）")

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
        line=dict(color='#2196F3', width=2.2, shape='hv'),
        hovertemplate='t: %{x:.2f}<br>总数: %{y}<extra></extra>'
    ), row=1, col=1)

    vx, vy = make_step_xy(valid_arrivals, T_shift)
    fig_cc2.add_trace(go.Scatter(
        x=vx, y=vy, mode='lines', name='有效来电',
        line=dict(color='#4CAF50', width=2.5, shape='hv'),
        hovertemplate='t: %{x:.2f}<br>有效: %{y}<extra></extra>'
    ), row=2, col=1)

    if len(valid_arrivals) > 0:
        fig_cc2.add_trace(go.Scatter(
            x=valid_arrivals, y=list(range(1, len(valid_arrivals) + 1)),
            mode='markers',
            marker=dict(size=5, color='#4CAF50', line=dict(width=0)),
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
            marker=dict(size=6, color='#F44336', symbol='x', line=dict(width=1.2)),
            name='骚扰电话',
            hovertemplate='骚扰: %{x:.3f}<extra></extra>'
        ), row=2, col=1)

    fig_cc2.update_xaxes(title_text=f'时间 (小时，共{T_shift}h)', row=2, col=1)
    fig_cc2.update_yaxes(title_text='累计来电', row=1, col=1)
    fig_cc2.update_yaxes(title_text='累计有效来电', row=2, col=1)
    fig_cc2.update_layout(
        height=600, template='plotly_white', hovermode='x unified',
        title=dict(text="骚扰电话过滤效果", font=dict(size=16)),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    for i in range(1, 3):
        fig_cc2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', row=i, col=1)
        fig_cc2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', row=i, col=1)
    st.plotly_chart(fig_cc2, use_container_width=True)

    # ── 运营指标 ──
    st.markdown("### 运营指标计算")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("每小时平均来电数", f"{len(all_arrivals) / T_shift:.2f}")
    c2.metric(
        "有效来电占比",
        f"{len(valid_arrivals) / max(len(all_arrivals), 1):.1%}"
    )
    c3.metric(
        "平均到达间隔",
        f"{T_shift / max(len(all_arrivals), 1):.2f} h" if len(all_arrivals) > 0 else "N/A"
    )
    c4.metric("有效 λ_eff", f"{effective_lam:.2f}/h")

    # 验证稀释性
    est_eff_lam = len(valid_arrivals) / T_shift
    st.markdown("**稀释性验证：**")
    cv1, cv2, cv3 = st.columns(3)
    cv1.metric("理论有效强度 λ(1-s)", f"{effective_lam:.2f}/h")
    cv2.metric("模拟有效强度", f"{est_eff_lam:.2f}/h")
    err_eff = abs(est_eff_lam - effective_lam) / effective_lam if effective_lam > 0 else 0
    cv3.markdown(
        f"相对误差 <span class='{get_error_color(err_eff)[1]}'>{err_eff:.2%}</span>",
        unsafe_allow_html=True
    )

    # ── 思考与结论 ──
    st.markdown(f"""
    <div class="conclusion-box">
    <b>📝 问题思考与结论</b><br>
    ① <b>排班优化：</b> {shift.split('(')[0]}理论来电强度为 {total_lam}/h，
    有效来电强度为 {effective_lam:.2f}/h。
    可根据有效来电强度合理配置坐席数，使服务率大于到达率，保证队列稳定。<br>
    ② <b>叠加性应用：</b> {n_lines} 条独立线路的来电叠加为强度 {total_lam}/h 的总泊松过程，
    验证了独立泊松流合并定理。<br>
    ③ <b>稀释性应用：</b> 以 {1-spam_ratio:.0%} 比例过滤骚扰电话后，
    有效来电仍为泊松过程（强度 {effective_lam:.2f}/h），
    这简化了排队系统中有效服务需求的分析。<br>
    ④ <b>资源配置价值：</b> 泊松过程理论可给出等待时间分布、溢出概率等关键指标，
    为客服中心科学排班提供数学基础。
    </div>
    """)


# ============================================================
# 主函数
# ============================================================

def main():
    st.title("泊松过程交互式可视化")
    st.markdown("《概率论与随机过程》期中作业  |  基本性质 · 叠加性 · 稀释性 · 客服中心应用")

    # ── 侧边栏：所有参数控件 ──
    with st.sidebar:
        st.header("控制面板")

        with st.expander("📊 基本性质参数", expanded=True):
            lam_basic = st.selectbox(
                "强度 λ", [0.5, 1.0, 2.0, 5.0], index=1,
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
            st.button("🔄 重新生成所有样本", key="basic_regenerate")

        st.markdown("---")

        with st.expander("🔗 叠加性参数", expanded=False):
            lam1_sup = st.slider("λ₁", 0.1, 5.0, 1.0, 0.1, key="sup_lam1")
            lam2_sup = st.slider("λ₂", 0.1, 5.0, 1.5, 0.1, key="sup_lam2")
            T_sup = st.slider("时间范围 T", 10.0, 100.0, 50.0, 5.0, key="sup_T")
            st.button("🔄 生成叠加过程", key="sup_generate")

        st.markdown("---")

        with st.expander("🎯 稀释性参数", expanded=False):
            lam_thin = st.slider("原始 λ", 0.1, 5.0, 2.0, 0.1, key="thin_lam")
            p_thin = st.slider("稀释概率 p", 0.0, 1.0, 0.6, 0.05, key="thin_p")
            T_thin = st.slider("时间范围 T", 10.0, 100.0, 50.0, 5.0, key="thin_T")
            st.button("🔄 生成稀释过程", key="thin_generate")

        st.markdown("---")

        with st.expander("📞 客服中心参数", expanded=False):
            shift_cc = st.selectbox(
                "时段选择",
                ['早班(9-12点, λ=2)', '午班(12-18点, λ=3.5)', '晚班(18-24点, λ=1.5)'],
                index=1, key="cc_shift"
            )
            lines_cc = st.slider("客服线路数", 1, 5, 3, 1, key="cc_lines")
            spam_cc = st.slider("骚扰电话比例", 0.0, 0.5, 0.2, 0.05, key="cc_spam")
            st.button("🔄 开始模拟", key="cc_simulate")

    # ── 标签页 ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 基本性质",
        "🔗 叠加性",
        "🎯 稀释性",
        "📞 客服中心应用"
    ])

    with tab1:
        render_tab_basic(lam_basic, T_basic, n_basic)

    with tab2:
        render_tab_superposition(lam1_sup, lam2_sup, T_sup)

    with tab3:
        render_tab_thinning(lam_thin, p_thin, T_thin)

    with tab4:
        render_tab_call_center(shift_cc, lines_cc, spam_cc)


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════
# 运行与导出说明
# ═══════════════════════════════════════════════════════════
#
# 安装依赖：
#   pip install streamlit numpy plotly scipy pandas
#
# 运行命令：
#   streamlit run app.py
#
# 导出 HTML（独立交互文件）：
#   1. 运行 streamlit run app.py
#   2. 浏览器中打开 http://localhost:8501
#   3. 点击右上角 ⋮ → Print → 目标选择 "另存为 HTML"
#   4. 保存后双击 .html 文件即可在浏览器中打开
#      (Plotly 图表保持完整交互：缩放、平移、悬停提示等)
#
# 依赖版本：
#   streamlit>=1.30.0, numpy>=1.24.0, plotly>=5.18.0,
#   scipy>=1.11.0, pandas>=2.0.0
# ═══════════════════════════════════════════════════════════
