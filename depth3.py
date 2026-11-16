import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 0. ページ初期設定（線アイコン指定・サイドバー展開）
# ==========================================
st.set_page_config(
    page_title="橋脚局所洗掘 時間発展予測・実験検証システム",
    page_icon=":material/waves:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# サイドバーの横幅ドラッグ調整 & ラジオボタン大型化CSS
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"][aria-expanded="true"] {
        resize: horizontal;
        overflow-x: auto;
        min-width: 260px;
        max-width: 720px;
    }
    div.stRadio > div[role="radiogroup"] {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False
}

G = 9.80665  # 重力加速度 [m/s^2]
LMIN_TO_M3S = 1.0 / 60000.0  # [l/min] -> [m^3/s] 換算係数


# ==========================================
# 1. 水理・洗掘・流砂量 共通コアエンジン関数群 (Ver. 3.2)
# ==========================================
def calc_kinematic_viscosity(temp_c: float) -> float:
    t_clipped = min(max(temp_c, 0.0), 40.0)
    return 1.78e-6 / (1.0 + 0.0337 * t_clipped + 0.000221 * (t_clipped ** 2))

def solve_normal_depth_manning(Q_m3s: float, B: float, I: float, n: float) -> float:
    if Q_m3s <= 0.0 or B <= 0.0 or I <= 0.0 or n <= 0.0:
        return 0.0
    def q_residual(y: float) -> float:
        A = B * y
        P = B + 2.0 * y
        R = A / P
        return (1.0 / n) * A * (R ** (2.0 / 3.0)) * (I ** 0.5) - Q_m3s
    y_low, y_high = 1e-5, 50.0
    for _ in range(100):
        y_mid = 0.5 * (y_low + y_high)
        if q_residual(y_mid) < 0.0:
            y_low = y_mid
        else:
            y_high = y_mid
    return 0.5 * (y_low + y_high)

def calc_critical_velocity_and_shields(d50_m: float, y: float, s: float, temp_c: float) -> tuple[float, float]:
    if y <= 0.0 or d50_m <= 0.0:
        return 0.0, 0.0
    nu = calc_kinematic_viscosity(temp_c)
    d_star = d50_m * (((s - 1.0) * G) / (nu ** 2)) ** (1.0 / 3.0)
    theta_cr = (0.30 / (1.0 + 1.2 * d_star)) + 0.055 * (1.0 - np.exp(-0.020 * d_star))
    u_star_c = np.sqrt(theta_cr * (s - 1.0) * G * d50_m)
    log_term = max(5.53 * (y / d50_m), 1.5)
    Vc = 5.75 * u_star_c * np.log10(log_term)
    return float(max(0.0, Vc)), float(max(0.0, theta_cr))

def calc_equilibrium_sediment_feed(
    V: float, Vc: float, theta_cr: float, B: float, d50_m: float, s: float,
    porosity: float = 0.40, bedload_formula: str = "芦田・道上 (1972) 式", feed_correction: float = 1.0
) -> dict:
    if V <= Vc or Vc <= 0.0 or theta_cr <= 0.0 or B <= 0.0 or d50_m <= 0.0:
        return {
            "tau_star": 0.0, "theta_cr": theta_cr, "phi": 0.0,
            "qs_g_s": 0.0, "qs_kg_hr": 0.0, "qs_mL_s": 0.0, "qs_m3_s": 0.0,
        }

    v_vc = V / Vc
    tau_star = theta_cr * (v_vc ** 2)

    if tau_star <= theta_cr:
        phi = 0.0
    elif "Meyer-Peter" in bedload_formula:
        phi = 8.0 * ((tau_star - theta_cr) ** 1.5)
    elif "修正" in bedload_formula or "Wong" in bedload_formula:
        phi = 3.97 * ((tau_star - theta_cr) ** 1.5)
    else:
        ratio = theta_cr / tau_star
        phi = 17.0 * (tau_star ** 1.5) * (1.0 - ratio) * (1.0 - np.sqrt(ratio))

    phi = float(max(0.0, phi))
    qb = phi * np.sqrt((s - 1.0) * G * (d50_m ** 3))
    
    # 補正係数 (feed_correction) を適用
    qs_m3_s = qb * B * feed_correction

    qs_kg_s = qs_m3_s * (s * 1000.0)
    qs_g_s = qs_kg_s * 1000.0
    qs_kg_hr = qs_kg_s * 3600.0

    solid_fraction = max(1.0 - porosity, 0.1)
    qs_bulk_m3_s = qs_m3_s / solid_fraction
    qs_mL_s = qs_bulk_m3_s * 1.0e6  # m^3/s -> mL/s

    return {
        "tau_star": float(tau_star), "theta_cr": float(theta_cr), "phi": phi,
        "qs_g_s": float(qs_g_s), "qs_kg_hr": float(qs_kg_hr),
        "qs_mL_s": float(qs_mL_s), "qs_m3_s": float(qs_m3_s),
    }

def core_scour_engine(
    Q_m3s: float, y_input: float, use_manning: bool, b: float, L: float, theta_deg: float, shape: str,
    custom_Ks: float, B: float, I: float, n: float, d50_mm: float, s: float, sigma_g: float, temp_c: float,
    dse_method: str, manual_dse: float = 0.0, alpha_live: float = 25.0, include_transition_dip: bool = True,
    porosity: float = 0.40, bedload_formula: str = "芦田・道上 (1972) 式", feed_correction: float = 1.0
) -> dict:
    d50 = d50_mm / 1000.0
    y = solve_normal_depth_manning(Q_m3s, B, I, n) if use_manning else y_input
    depth_source = "マニング等流水深" if use_manning else "実測・設定水深"

    if y <= 0.0 or Q_m3s <= 0.0:
        return {
            "y": y, "depth_source": depth_source, "V": 0.0, "Vc": 0.0, "V_Vc": 0.0,
            "Vlp": 0.0, "Vlp_Vc": 0.0, "Fr": 0.0, "dse": 0.0, "te_sec": 0.0,
            "Ks": 1.0, "K_theta": 1.0, "K_sigma": 1.0, "regime": "無水・流量ゼロ",
            "qs_g_s": 0.0, "qs_kg_hr": 0.0, "qs_mL_s": 0.0, "tau_star": 0.0, "theta_cr": 0.0
        }

    V = Q_m3s / (B * y)
    Fr = V / np.sqrt(G * y)
    Vc, theta_cr = calc_critical_velocity_and_shields(d50, y, s, temp_c)
    V_Vc = V / Vc if Vc > 0.0 else 0.0

    feed = calc_equilibrium_sediment_feed(V, Vc, theta_cr, B, d50, s, porosity, bedload_formula, feed_correction)

    Vlp = max(5.0 * Vc, 0.6 * np.sqrt(G * y))
    Vlp_Vc = Vlp / Vc if Vc > 0.0 else 5.0

    if V_Vc <= 0.4:
        return {
            "y": float(y), "depth_source": depth_source, "V": float(V), "Vc": float(Vc),
            "V_Vc": float(V_Vc), "Vlp": float(Vlp), "Vlp_Vc": float(Vlp_Vc), "Fr": float(Fr),
            "dse": 0.0, "te_sec": 0.0, "Ks": 1.0, "K_theta": 1.0, "K_sigma": 1.0,
            "regime": "洗掘なし (V/Vc <= 0.4)",
            "qs_g_s": 0.0, "qs_kg_hr": 0.0, "qs_mL_s": 0.0, "tau_star": feed["tau_star"], "theta_cr": theta_cr
        }

    if V_Vc <= 1.0: regime = "静的洗掘 (Clear-water)"
    elif V_Vc <= Vlp_Vc: regime = "動的洗掘 (Live-bed)"
    else: regime = "動的洗掘・平坦河床域 (V > Vlp)"

    Ks = custom_Ks if shape == "任意係数指定 (Custom)" else (1.1 if shape == "矩形 (Rectangular)" else 1.0)
    rad = np.radians(theta_deg)
    K_theta = 1.0 if (shape == "円柱 (Circular)" or theta_deg == 0.0) else ((L / b) * np.sin(rad) + np.cos(rad)) ** 0.65
    K_sigma = 1.0 if sigma_g <= 1.3 else max(0.5, 1.0 - 0.25 * np.log10(sigma_g / 1.3))

    if dse_method == "手入力": dse = max(0.0, manual_dse)
    elif dse_method == "Melville & Coleman (2000) 法":
        y_b = y / b
        K_yD = 2.4 * b if y_b < 0.7 else (2.0 * np.sqrt(y * b) if y_b < 5.0 else 4.5 * b)
        if V_Vc <= 1.0: K_I = max(V_Vc, 0.0)
        else: K_I = (1.0 - 0.14 * np.exp(-((V_Vc - 1.30) / 0.28) ** 2)) if (include_transition_dip and V_Vc < Vlp_Vc) else 1.0
        b_d50 = b / d50
        K_d = max(0.0, 0.57 * np.log10(2.24 * b_d50)) if b_d50 < 25.0 else 1.0
        dse = max(0.0, K_yD * K_I * K_d * Ks * K_theta * K_sigma)
    else:
        y_star = y / b
        d_star_ratio = b / d50
        f1 = np.tanh(y_star ** 0.4)
        f3 = d_star_ratio / (0.4 * (d_star_ratio ** 1.2) + 10.6 * (d_star_ratio ** -0.13))
        if V_Vc <= 1.0:
            V_star = max(V_Vc, 0.4)
            dse_cyl = 2.5 * b * f1 * max(0.0, 1.0 - 1.2 * (np.log(V_star) ** 2)) * f3
        elif V_Vc <= Vlp_Vc:
            w = (V_Vc - 1.0) / max(Vlp_Vc - 1.0, 1e-6)
            dse_linear = (2.5 * b * f1 * f3) * (1.0 - w) + (2.2 * b * f1) * w
            dse_cyl = dse_linear * (1.0 - 0.14 * np.exp(-((V_Vc - 1.30) / 0.28) ** 2)) if include_transition_dip else dse_linear
        else:
            dse_cyl = 2.2 * b * f1
        dse = max(0.0, dse_cyl * Ks * K_theta * K_sigma)

    y_b = y / b
    if V_Vc <= 1.0:
        te_days = 30.89 * (b / V) * (V_Vc - 0.4) * (y_b ** 0.25 if y_b <= 6.0 else 1.0) if y_b > 6.0 else 30.89 * (b / V) * (V_Vc - 0.4) * (y_b ** 0.25)
        if y_b > 6.0: te_days = 48.26 * (b / V) * (V_Vc - 0.4)
    else:
        te_1_days = 30.89 * (b / Vc) * 0.6 * (y_b ** 0.25) if y_b <= 6.0 else 48.26 * (b / Vc) * 0.6
        te_days = te_1_days / (1.0 + alpha_live * ((V_Vc - 1.0) ** 1.5))
    
    te_sec = max(te_days * 86400.0, 1e-6)

    return {
        "y": float(y), "depth_source": depth_source, "V": float(V), "Vc": float(Vc),
        "V_Vc": float(V_Vc), "Vlp": float(Vlp), "Vlp_Vc": float(Vlp_Vc), "Fr": float(Fr),
        "dse": float(dse), "te_sec": float(te_sec), "Ks": float(Ks), "K_theta": float(K_theta),
        "K_sigma": float(K_sigma), "regime": regime,
        "qs_g_s": feed["qs_g_s"], "qs_kg_hr": feed["qs_kg_hr"],
        "qs_mL_s": feed["qs_mL_s"], "tau_star": feed["tau_star"], "theta_cr": theta_cr,
    }

def calc_kt_melville_chiew(t_sec: float, te_sec: float, V_Vc: float) -> float:
    if t_sec <= 0.0 or te_sec <= 0.0 or V_Vc <= 0.4: return 0.0
    if t_sec >= te_sec: return 1.0
    return min(max(float(np.exp(-0.03 * (abs((1.0 / V_Vc) * np.log(t_sec / te_sec)) ** 1.6))), 0.0), 1.0)

def calc_equivalent_time_star(Kt_star: float, te_sec: float, V_Vc: float) -> float:
    if Kt_star <= 0.0 or te_sec <= 0.0 or V_Vc <= 0.4: return 0.0
    if Kt_star >= 1.0: return te_sec
    return float(te_sec * np.exp(-V_Vc * ((-np.log(min(max(Kt_star, 1e-12), 1.0 - 1e-12)) / 0.03) ** (1.0 / 1.6))))


# ==========================================
# 2. タイトル & 大型モード切替ラジオボタン
# ==========================================
st.title(":material/waves: 橋脚局所洗掘 時間発展予測・実験検証システム")
st.caption("Ver. 3.2 ｜ 静的洗掘＆動的洗掘統合解析 ＋ 実用給砂レート [g/s] 補正 ＋ 実験誤差検証")

# 選択モードが分かりやすい大型ラジオボタンとアラートバナー
mode = st.radio(
    "📝 計算モードの選択",
    ["定常流モード (Steady)", "非定常流モード (Unsteady)"],
    horizontal=True,
    label_visibility="collapsed"
)
is_unsteady = "非定常" in mode

if not is_unsteady:
    st.success("🟢 **現在【定常流モード (Steady)】が選択されています。** 一定流量での時間発展を計算します。")
else:
    st.info("🔵 **現在【非定常流モード (Unsteady)】が選択されています。** ハイドログラフを入力して水位・流量変化を追跡します。")


# ==========================================
# 3. サイドバー：共通諸元の入力UI
# ==========================================
with st.sidebar:
    st.header(":material/architecture: 1. 橋脚諸元")
    pier_shape = st.selectbox("橋脚の平面形状", ["小判型 (Round-nosed)", "円柱 (Circular)", "矩形 (Rectangular)", "任意係数指定 (Custom)"], index=0)
    custom_Ks = st.number_input("形状係数 Ks [-]", min_value=0.1, max_value=3.0, value=1.0, step=0.05, format="%.2f") if pier_shape == "任意係数指定 (Custom)" else 1.0
    col_p1, col_p2 = st.columns(2)
    with col_p1: pier_b = st.number_input("橋脚幅 b [m]", min_value=0.005, max_value=20.0, value=0.05, step=0.01, format="%.3f")
    with col_p2: pier_L = st.number_input("橋脚長さ L [m]", min_value=0.005, max_value=100.0, value=0.15, step=0.01, format="%.3f")
    theta_deg = st.number_input("流れに対する迎え角 θ [deg]", min_value=0.0, max_value=90.0, value=0.0, step=1.0)

    st.divider()
    st.header(":material/straighten: 2. 水路・河道諸元")
    col_c1, col_c2 = st.columns(2)
    with col_c1: chan_B = st.number_input("水路幅 B [m]", min_value=0.05, max_value=500.0, value=0.50, step=0.05, format="%.3f")
    with col_c2: chan_L = st.number_input("水路長さ Lch [m]", min_value=0.1, max_value=5000.0, value=10.0, step=0.5, format="%.2f")
    depth_mode = st.radio("水深 y の決定方法（区別必須）", ["実測・設定水深 y を直接入力", "勾配 I と粗度 n から等流水深を逆算"], index=0)
    use_manning = (depth_mode == "勾配 I と粗度 n から等流水深を逆算")
    col_c3, col_c4 = st.columns(2)
    with col_c3: slope_I = st.number_input("河床勾配 I [-]", min_value=0.00001, max_value=0.5, value=0.001, step=0.0005, format="%.5f", disabled=not use_manning)
    with col_c4: manning_n = st.number_input("マニング粗度 n", min_value=0.005, max_value=0.100, value=0.015, step=0.001, format="%.3f", disabled=not use_manning)

    st.divider()
    st.header(":material/layers: 3. 河床材料・水温")
    col_s1, col_s2 = st.columns(2)
    with col_s1: d50_mm = st.number_input("代表粒径 d50 [mm]", min_value=0.05, max_value=200.0, value=0.80, step=0.10, format="%.2f")
    with col_s2: sed_s = st.number_input("砂粒子比重 s [-]", min_value=1.10, max_value=5.00, value=2.65, step=0.05, format="%.2f")
    col_s3, col_s4 = st.columns(2)
    with col_s3: sigma_g = st.number_input("幾何標準偏差 σg [-]", min_value=1.00, max_value=5.00, value=1.20, step=0.05, format="%.2f")
    with col_s4: temp_c = st.number_input("水温 T [℃] (0〜40℃)", min_value=0.0, max_value=40.0, value=20.0, step=1.0, format="%.1f")

    st.divider()
    st.header(":material/calculate: 4. 平衡洗掘深 dse モデル")
    dse_method = st.selectbox("dse 算定モデル", ["Sheppard / Melville (S/M) 法", "Melville & Coleman (2000) 法", "手入力"], index=0)
    manual_dse_mm = st.number_input("指定する平衡洗掘深 dse [mm]", min_value=0.0, max_value=5000.0, value=100.0, step=5.0) if dse_method == "手入力" else 0.0

    st.divider()
    with st.expander("5. 動的洗掘・給砂レート詳細設定", expanded=True, icon=":material/tune:"):
        st.markdown("給砂レートの算定と補正")
        bedload_formula = st.selectbox("給砂レート算定式", ["芦田・道上 (1972) 式", "Meyer-Peter & Müller (1948) 式", "修正 MPM 式 (Wong & Parker, 2006)"], index=0)
        feed_correction = st.slider(
            "給砂量 補正係数 [-]", min_value=0.1, max_value=2.0, value=1.0, step=0.1,
            help="理論値が実測より多い場合、0.5 などに下げて実験装置での適正量にスケーリングします。"
        )
        sed_porosity = st.slider("砂の空隙率 λ [-]（かさ体積換算用）", min_value=0.25, max_value=0.55, value=0.40, step=0.02)
        st.divider()
        alpha_live = st.slider("平衡時間短縮係数 α [-]", min_value=5.0, max_value=60.0, value=25.0, step=5.0)
        include_transition_dip = st.checkbox("遷移域の一時低下（ディップ）補正を反映", value=True)
        show_bedform_band = st.checkbox("河床波（デューン）変動バンドを表示", value=True)
        bedform_amp_pct = st.slider("河床波による上下変動幅 [±%]", min_value=5, max_value=30, value=15, step=5, disabled=not show_bedform_band)
        enable_refilling = st.checkbox("非定常減水期の埋め戻し（Refilling）を有効化", value=True)


# ==========================================
# 4. 実験条件の自動診断アラート
# ==========================================
blockage_ratio = chan_B / pier_b
if blockage_ratio < 8.0:
    st.warning(f"**側壁影響（閉塞効果）警告**：水路幅/橋脚幅比 $B/b = {blockage_ratio:.1f} < 8.0$ です。洗掘深が過大になる恐れがあります。", icon=":material/warning:")
elif blockage_ratio < 10.0:
    st.info(f"**側壁影響の注意**：$B/b = {blockage_ratio:.1f}$（推奨目安は $B/b \\ge 8 \\sim 10$）です。", icon=":material/info:")


# ==========================================
# 5. モード別の水理条件入力 & 計算実行 & 描画
# ==========================================
if not is_unsteady:
    # ------------------------------------------
    # 定常流モード (Steady)
    # ------------------------------------------
    st.subheader(":material/trending_flat: 定常流モード：通水条件の設定")
    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        Q_steady_lmin = st.number_input("流量 Q [l/min]", min_value=0.1, max_value=300000000.0, value=1200.0, step=50.0, format="%.1f")
        Q_steady_m3s = Q_steady_lmin * LMIN_TO_M3S
    with col_q2:
        if not use_manning:
            y_steady = st.number_input("接近流水深 y [m]（実測値）", min_value=0.005, max_value=50.0, value=0.150, step=0.010, format="%.3f")
        else:
            y_steady = solve_normal_depth_manning(Q_steady_m3s, chan_B, slope_I, manning_n)
            st.metric("マニング等流水深 y（自動算定）", f"{y_steady:.4f} m")
    with col_q3:
        t_end_min = st.number_input("通水時間 t_end [min]", min_value=1.0, max_value=60000.0, value=360.0, step=10.0, format="%.1f")

    if Q_steady_lmin <= 0.0 or y_steady <= 0.0: st.stop()

    res = core_scour_engine(
        Q_steady_m3s, y_steady, use_manning, pier_b, pier_L, theta_deg, pier_shape, custom_Ks,
        chan_B, slope_I, manning_n, d50_mm, sed_s, sigma_g, temp_c, dse_method, manual_dse_mm / 1000.0,
        alpha_live, include_transition_dip, sed_porosity, bedload_formula, feed_correction
    )

    t_end_sec = t_end_min * 60.0
    t_sec_arr = np.unique(np.concatenate(([0.0], np.geomspace(1.0, max(t_end_sec, 2.0), 150), np.linspace(0.0, t_end_sec, 150))))
    t_sec_arr = t_sec_arr[t_sec_arr <= t_end_sec]

    Kt_arr = np.array([calc_kt_melville_chiew(t, res["te_sec"], res["V_Vc"]) for t in t_sec_arr])
    ds_m_arr = Kt_arr * res["dse"]

    df_steady = pd.DataFrame({
        "time_min": t_sec_arr / 60.0, "time_sec": t_sec_arr, "depth_source": res["depth_source"],
        "regime": res["regime"], "Q_l_min": Q_steady_lmin, "V_Vc": res["V_Vc"],
        "Qs_g_s": res["qs_g_s"], "Qs_mL_s": res["qs_mL_s"],
        "Kt_percent": Kt_arr * 100.0, "ds_mm": ds_m_arr * 1000.0, "dse_mm": res["dse"] * 1000.0
    })

    # KPI 1段目
    st.divider()
    st.subheader(f":material/analytics: 予測結果サマリー（{res['regime']}）")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(f"{t_end_min:.0f} min 後洗掘深 ds", f"{ds_m_arr[-1] * 1000.0:.1f} mm", f"到達率 {Kt_arr[-1] * 100.0:.1f} %")
    k2.metric("平衡洗掘深 dse", f"{res['dse']*1000.0:.1f} mm", f"ds/b = {res['dse']/pier_b:.2f}")
    k3.metric("平衡到達時間 te", f"{res['te_sec']/60.0:.0f} min", f"約 {res['te_sec']/3600.0:.1f} hr")
    k4.metric("流速強度 V / Vc", f"{res['V_Vc']:.3f}", f"V={res['V']:.3f} m/s")
    k5.metric("無次元水深 y/b", f"{res['y']/pier_b:.2f}", f"Fr = {res['Fr']:.3f}")

    # KPI 2段目（給砂レート g/s）
    with st.container(border=True):
        st.markdown(f"**:material/sand: 適正給砂レート目安（補正係数: {feed_correction}x ｜ 算定式: {bedload_formula}）**")
        s1, s2, s3, s4 = st.columns(4)
        total_sand_kg = (res["qs_g_s"] * t_end_sec) / 1000.0
        total_sand_L = (res["qs_mL_s"] * t_end_sec) / 1000.0

        if res["V_Vc"] > 1.0:
            s1.metric("質量給砂レート", f"{res['qs_g_s']:.2f} g/s", f"約 {res['qs_kg_hr']:.2f} kg/hr")
            s2.metric(f"かさ体積レート (λ={sed_porosity:.2f})", f"{res['qs_mL_s']:.1f} mL/s", f"約 {res['qs_mL_s'] * 0.06:.2f} L/min")
            s3.metric(f"通水 {t_end_min:.0f} min の必要総砂量", f"{total_sand_kg:.2f} kg", f"かさ体積 約 {total_sand_L:.2f} L")
            s4.metric("有効掃流力 τ*' / 限界 θcr", f"{res['tau_star']:.4f}", f"限界 θcr = {res['theta_cr']:.4f}")
        else:
            s1.metric("質量給砂レート", "0.00 g/s", "静的洗掘域のため給砂不要")
            s2.metric(f"かさ体積レート (λ={sed_porosity:.2f})", "0.0 mL/s", "0.0 L/min")
            s3.metric(f"通水 {t_end_min:.0f} min の必要総砂量", "0.00 kg", "0.00 L")
            s4.metric("有効掃流力 τ*' / 限界 θcr", f"{res['tau_star']:.4f}", f"限界 θcr = {res['theta_cr']:.4f}")

    # グラフ描画
    axis_type = st.radio("横軸（時間軸）の表示スケール", ["通常軸 (Linear)", "対数軸 (Log)"], horizontal=True)
    x_axis_mode = "log" if "対数" in axis_type else "linear"
    plot_df = df_steady[df_steady["time_min"] > 0] if x_axis_mode == "log" else df_steady

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if res["V_Vc"] > 1.0 and show_bedform_band and res["dse"] > 0.0:
        amp = bedform_amp_pct / 100.0
        fig.add_trace(go.Scatter(
            x=pd.concat([plot_df["time_min"], plot_df["time_min"][::-1]]),
            y=pd.concat([plot_df["ds_mm"] * (1.0 + amp), (plot_df["ds_mm"] * (1.0 - amp))[::-1]]),
            fill="toself", fillcolor="rgba(31, 119, 180, 0.16)", line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
            name=f"河床波による変動帯 (±{bedform_amp_pct}%)"
        ), secondary_y=False)
    fig.add_trace(go.Scatter(x=plot_df["time_min"], y=plot_df["ds_mm"], mode="lines", name="理論予測洗掘深 ds(t) [mm]", line=dict(color="#1f77b4", width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=plot_df["time_min"], y=plot_df["Kt_percent"], mode="lines", name="到達率 Kt [%]", line=dict(color="#ff7f0e", width=2, dash="dash")), secondary_y=True)

    fig.update_layout(
        title="洗掘深 ds(t) および到達率 Kt の時間発展曲線", xaxis_title="通水時間 t [min]", xaxis_type=x_axis_mode,
        hovermode="x unified", dragmode=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=20, r=20, t=60, b=20), height=440
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(title_text="洗掘深 ds [mm]", secondary_y=False, rangemode="tozero", fixedrange=True)
    fig.update_yaxes(title_text="平衡到達率 Kt [%]", secondary_y=True, range=[0, 105], fixedrange=True)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

else:
    # ------------------------------------------
    # 非定常流モード (Unsteady)
    # ------------------------------------------
    st.subheader(":material/show_chart: 非定常流モード：ハイドログラフの入力と追跡")
    
    col_u1, col_u2 = st.columns([1, 2])
    with col_u1:
        dt_sec = st.number_input("計算時間刻み Δt [sec]", min_value=1.0, max_value=600.0, value=10.0, step=5.0)
    with col_u2:
        default_hydro = pd.DataFrame({
            "time_min": [0.0, 20.0, 40.0, 60.0, 90.0, 120.0, 160.0, 200.0],
            "Q_l_min": [600.0, 1200.0, 2200.0, 2600.0, 2100.0, 1500.0, 900.0, 600.0],
            "y_m": [0.10, 0.14, 0.18, 0.20, 0.18, 0.15, 0.12, 0.10]
        })
        hydro_df = st.data_editor(
            default_hydro, num_rows="dynamic", use_container_width=True,
            column_config={
                "time_min": st.column_config.NumberColumn("時刻 time [min]", format="%.1f"),
                "Q_l_min": st.column_config.NumberColumn("流量 Q [l/min]", format="%.1f"),
                "y_m": st.column_config.NumberColumn("水深 y [m] (任意)", format="%.3f"),
            }
        )

    clean_hydro = hydro_df.dropna(subset=["time_min", "Q_l_min"]).sort_values("time_min").drop_duplicates("time_min").reset_index(drop=True)
    if len(clean_hydro) < 2: st.stop()

    t_raw_sec = clean_hydro["time_min"].values.astype(float) * 60.0
    q_raw_m3s = clean_hydro["Q_l_min"].values.astype(float) * LMIN_TO_M3S
    y_raw_list = [float(y_val) if not use_manning and pd.notna(y_val) and float(y_val) > 0.0 else solve_normal_depth_manning(float(q_val), chan_B, slope_I, manning_n) for q_val, y_val in zip(q_raw_m3s, clean_hydro["y_m"])]
    y_raw = np.array(y_raw_list, dtype=float)

    t_start, t_end = t_raw_sec[0], t_raw_sec[-1]
    n_steps = max(int(np.ceil((t_end - t_start) / dt_sec)), 1)
    t_grid = np.linspace(t_start, t_end, n_steps + 1)
    q_grid = np.interp(t_grid, t_raw_sec, q_raw_m3s)
    y_grid = np.interp(t_grid, t_raw_sec, y_raw)

    records, ds_prev, cum_sand_g, cum_sand_L = [], 0.0, 0.0, 0.0

    for i in range(len(t_grid)):
        if i == 0:
            res_i = core_scour_engine(q_grid[0], y_grid[0], False, pier_b, pier_L, theta_deg, pier_shape, custom_Ks, chan_B, slope_I, manning_n, d50_mm, sed_s, sigma_g, temp_c, dse_method, manual_dse_mm / 1000.0, alpha_live, include_transition_dip, sed_porosity, bedload_formula, feed_correction)
            ds_curr = 0.0
        else:
            dt_step = t_grid[i] - t_grid[i - 1]
            q_bar = 0.5 * (q_grid[i - 1] + q_grid[i])
            y_bar = 0.5 * (y_grid[i - 1] + y_grid[i])
            res_i = core_scour_engine(q_bar, y_bar, False, pier_b, pier_L, theta_deg, pier_shape, custom_Ks, chan_B, slope_I, manning_n, d50_mm, sed_s, sigma_g, temp_c, dse_method, manual_dse_mm / 1000.0, alpha_live, include_transition_dip, sed_porosity, bedload_formula, feed_correction)
            
            cum_sand_g += res_i["qs_g_s"] * dt_step
            cum_sand_L += (res_i["qs_mL_s"] / 1000.0) * dt_step

            if res_i["dse"] <= 0.0 or res_i["te_sec"] <= 0.0:
                ds_curr = ds_prev
            elif ds_prev >= res_i["dse"]:
                ds_curr = res_i["dse"] + (ds_prev - res_i["dse"]) * np.exp(-3.0 * dt_step / max(res_i["te_sec"], dt_step)) if enable_refilling and res_i["V_Vc"] > 1.0 else ds_prev
            else:
                t_star = calc_equivalent_time_star(ds_prev / res_i["dse"], res_i["te_sec"], res_i["V_Vc"])
                kt_next = calc_kt_melville_chiew(t_star + dt_step, res_i["te_sec"], res_i["V_Vc"])
                ds_curr = max(ds_prev, kt_next * res_i["dse"])

        ds_prev = ds_curr
        records.append({
            "time_min": t_grid[i] / 60.0, "Q_l_min": q_grid[i] / LMIN_TO_M3S, "y_m": res_i["y"],
            "V_Vc": res_i["V_Vc"], "Qs_g_s": res_i["qs_g_s"], "Qs_mL_s": res_i["qs_mL_s"],
            "dse_mm": res_i["dse"] * 1000.0, "ds_mm": ds_curr * 1000.0
        })

    df_unsteady = pd.DataFrame(records)

    st.divider()
    st.subheader(":material/analytics: 非定常流シミュレーション結果サマリー")
    uk1, uk2, uk3, uk4 = st.columns(4)
    uk1.metric("最大到達洗掘深 ds,max", f"{df_unsteady['ds_mm'].max():.1f} mm")
    uk2.metric("通水終了時洗掘深 ds,end", f"{df_unsteady['ds_mm'].iloc[-1]:.1f} mm")
    uk3.metric("ピーク流速強度 (V/Vc)max", f"{df_unsteady['V_Vc'].max():.3f}")
    uk4.metric("全期間総必要砂量 (質量 / 体積)", f"{cum_sand_g/1000.0:.2f} kg", f"かさ体積 {cum_sand_L:.2f} L")

    fig_u = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": False}]], subplot_titles=("1. ハイドログラフ Q と流速強度 V/Vc", f"2. 給砂レート推移 [g/s]（補正 {feed_correction}x）", "3. 瞬間平衡洗掘深 dse(t) と累積洗掘深 ds(t)"))
    fig_u.add_trace(go.Scatter(x=df_unsteady["time_min"], y=df_unsteady["Q_l_min"], mode="lines", name="流量 Q [l/min]", line=dict(color="#1f77b4", width=2.5)), row=1, col=1, secondary_y=False)
    fig_u.add_trace(go.Scatter(x=df_unsteady["time_min"], y=df_unsteady["V_Vc"], mode="lines", name="V/Vc [-]", line=dict(color="#ff7f0e", width=2, dash="dash")), row=1, col=1, secondary_y=True)
    fig_u.add_hline(y=1.0, line_dash="dot", line_color="gray", row=1, col=1, secondary_y=True)

    fig_u.add_trace(go.Scatter(x=df_unsteady["time_min"], y=df_unsteady["Qs_g_s"], mode="lines", name="給砂 [g/s]", line=dict(color="#8c564b", width=2.2)), row=2, col=1, secondary_y=False)
    fig_u.add_trace(go.Scatter(x=df_unsteady["time_min"], y=df_unsteady["Qs_mL_s"], mode="lines", name="かさ体積 [mL/s]", line=dict(color="#17becf", width=1.8, dash="dot")), row=2, col=1, secondary_y=True)

    if show_bedform_band:
        amp = bedform_amp_pct / 100.0
        upper, lower = np.where(df_unsteady["V_Vc"] > 1.0, df_unsteady["ds_mm"]*(1+amp), df_unsteady["ds_mm"]), np.where(df_unsteady["V_Vc"] > 1.0, df_unsteady["ds_mm"]*(1-amp), df_unsteady["ds_mm"])
        fig_u.add_trace(go.Scatter(x=pd.concat([df_unsteady["time_min"], df_unsteady["time_min"][::-1]]), y=np.concatenate([upper, lower[::-1]]), fill="toself", fillcolor="rgba(44, 160, 44, 0.16)", line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", name=f"変動帯 (±{bedform_amp_pct}%)"), row=3, col=1)
    fig_u.add_trace(go.Scatter(x=df_unsteady["time_min"], y=df_unsteady["dse_mm"], mode="lines", name="dse(t) [mm]", line=dict(color="#d62728", width=1.8, dash="dash")), row=3, col=1)
    fig_u.add_trace(go.Scatter(x=df_unsteady["time_min"], y=df_unsteady["ds_mm"], mode="lines", name="ds(t) [mm]", line=dict(color="#2ca02c", width=3)), row=3, col=1)

    fig_u.update_layout(hovermode="x unified", dragmode=False, legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1), margin=dict(l=20, r=20, t=80, b=20), height=760)
    fig_u.update_xaxes(fixedrange=True)
    fig_u.update_yaxes(title_text="Q [l/min]", row=1, col=1, secondary_y=False, rangemode="tozero", fixedrange=True)
    fig_u.update_yaxes(title_text="給砂 [g/s]", row=2, col=1, secondary_y=False, rangemode="tozero", fixedrange=True)
    fig_u.update_yaxes(title_text="体積 [mL/s]", row=2, col=1, secondary_y=True, rangemode="tozero", fixedrange=True)
    fig_u.update_yaxes(title_text="洗掘深 [mm]", row=3, col=1, rangemode="tozero", fixedrange=True)
    st.plotly_chart(fig_u, use_container_width=True, config=PLOTLY_CONFIG)