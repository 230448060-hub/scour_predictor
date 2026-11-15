import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 0. ページ初期設定（線アイコン指定）
# ==========================================
st.set_page_config(
    page_title="橋脚局所洗掘 時間発展予測・実験検証システム",
    page_icon=":material/waves:",
    layout="wide"
)

G = 9.80665  # 重力加速度 [m/s^2]
LMIN_TO_M3S = 1.0 / 60000.0  # [l/min] -> [m^3/s] 換算係数

# ==========================================
# 1. 水理・洗掘 共通コアエンジン関数群
# ==========================================
def calc_kinematic_viscosity(temp_c: float) -> float:
    """
    Poiseuilleの経験式により水温[℃]から水の動粘性係数 ν [m^2/s] を算出
    適用範囲: 0 ℃ <= T <= 40 ℃
    """
    t_clipped = min(max(temp_c, 0.0), 40.0)
    return 1.78e-6 / (1.0 + 0.0337 * t_clipped + 0.000221 * (t_clipped ** 2))


def solve_normal_depth_manning(Q_m3s: float, B: float, I: float, n: float) -> float:
    """長方形断面のマニング公式を二分法で解き、等流水深 y [m] を返す（内部SI単位: Q[m^3/s]）"""
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


def calc_critical_velocity(d50_m: float, y: float, s: float, temp_c: float) -> float:
    """Soulsbyの限界シールズ数式＋対数流速分布則により移動限界平均流速 Vc [m/s] を算出"""
    if y <= 0.0 or d50_m <= 0.0:
        return 0.0
    nu = calc_kinematic_viscosity(temp_c)
    d_star = d50_m * (((s - 1.0) * G) / (nu ** 2)) ** (1.0 / 3.0)
    theta_cr = (0.30 / (1.0 + 1.2 * d_star)) + 0.055 * (1.0 - np.exp(-0.020 * d_star))
    u_star_c = np.sqrt(theta_cr * (s - 1.0) * G * d50_m)
    log_term = max(5.53 * (y / d50_m), 1.5)
    Vc = 5.75 * u_star_c * np.log10(log_term)
    return float(max(0.0, Vc))


def core_scour_engine(
    Q_m3s: float,
    y_input: float,
    use_manning: bool,
    b: float,
    L: float,
    theta_deg: float,
    shape: str,
    custom_Ks: float,
    B: float,
    I: float,
    n: float,
    d50_mm: float,
    s: float,
    sigma_g: float,
    temp_c: float,
    dse_method: str,
    manual_dse: float = 0.0
) -> dict:
    """
    【共通コアエンジン】
    1時点の水理条件（Q_m3s [m^3/s]）から、V, Vc, 平衡洗掘深 dse [m], 平衡到達時間 te [sec] を算出する
    """
    d50 = d50_mm / 1000.0  # [mm] -> [m]
    y = solve_normal_depth_manning(Q_m3s, B, I, n) if use_manning else y_input
    depth_source = "マニング等流水深" if use_manning else "実測・設定水深"

    if y <= 0.0 or Q_m3s <= 0.0:
        return {
            "y": y, "depth_source": depth_source, "V": 0.0, "Vc": 0.0, "V_Vc": 0.0, "Fr": 0.0,
            "dse": 0.0, "te_sec": 0.0, "Ks": 1.0, "K_theta": 1.0, "K_sigma": 1.0,
            "regime": "無水・流量ゼロ"
        }

    V = Q_m3s / (B * y)
    Fr = V / np.sqrt(G * y)
    Vc = calc_critical_velocity(d50, y, s, temp_c)
    V_Vc = V / Vc if Vc > 0.0 else 0.0

    # 洗掘発生限界の判定 (V/Vc <= 0.4 は洗掘なし: Melville & Chiew, 1999)
    if V_Vc <= 0.4:
        return {
            "y": float(y), "depth_source": depth_source,
            "V": float(V), "Vc": float(Vc), "V_Vc": float(V_Vc), "Fr": float(Fr),
            "dse": 0.0, "te_sec": 0.0, "Ks": 1.0, "K_theta": 1.0, "K_sigma": 1.0,
            "regime": "洗掘なし (V/Vc <= 0.4)"
        }

    regime = "静的洗掘 (Clear-water)" if V_Vc <= 1.0 else "動的洗掘 (Live-bed / 範囲外警告)"
    V_eff = min(V, Vc)
    V_eff_Vc = V_eff / Vc

    # 形状係数 Ks (Melville & Coleman, 2000 基準 ＋ ユーザー上書き対応)
    if shape == "任意係数指定 (Custom)":
        Ks = custom_Ks
    elif shape == "矩形 (Rectangular)":
        Ks = 1.1
    else:
        Ks = 1.0  # 小判型 (Round-nosed)・円柱 (Circular)

    # 入射角係数 K_theta (Melville & Coleman, 2000)
    if shape == "円柱 (Circular)" or theta_deg == 0.0:
        K_theta = 1.0
    else:
        rad = np.radians(theta_deg)
        K_theta = ((L / b) * np.sin(rad) + np.cos(rad)) ** 0.65

    # 粒度分布係数 K_sigma (一様砂 sigma_g <= 1.3 なら 1.0)
    K_sigma = 1.0 if sigma_g <= 1.3 else max(0.5, 1.0 - 0.25 * np.log10(sigma_g / 1.3))

    # 最終平衡洗掘深 dse [m] の算定
    if dse_method == "手入力":
        dse = max(0.0, manual_dse)
    elif dse_method == "Melville & Coleman (2000) 法":
        y_b = y / b
        if y_b < 0.7:
            K_yD = 2.4 * b
        elif y_b < 5.0:
            K_yD = 2.0 * np.sqrt(y * b)
        else:
            K_yD = 4.5 * b
        K_I = min(max(V_Vc, 0.0), 1.0)
        b_d50 = b / d50
        K_d = max(0.0, 0.57 * np.log10(2.24 * b_d50)) if b_d50 < 25.0 else 1.0
        dse = max(0.0, K_yD * K_I * K_d * Ks * K_theta * K_sigma)
    else:
        # Sheppard / Melville (S/M) 法（負値発生防止済み）
        y_star = y / b
        d_star_ratio = b / d50
        V_star = min(max(V_Vc, 0.4), 1.0)
        f1 = np.tanh(y_star ** 0.4)
        f2 = max(0.0, 1.0 - 1.2 * (np.log(V_star) ** 2))
        f3 = d_star_ratio / (0.4 * (d_star_ratio ** 1.2) + 10.6 * (d_star_ratio ** -0.13))
        dse_cyl = 2.5 * b * f1 * f2 * f3
        dse = max(0.0, dse_cyl * Ks * K_theta * K_sigma)

    # Melville & Chiew (1999) 平衡到達時間 te [days -> sec]
    y_b = y / b
    if y_b <= 6.0:
        te_days = 30.89 * (b / V_eff) * (V_eff_Vc - 0.4) * (y_b ** 0.25)
    else:
        te_days = 48.26 * (b / V_eff) * (V_eff_Vc - 0.4)
    te_sec = max(te_days * 86400.0, 1e-6)

    # 事後異常値チェック (NaN / Inf / 負値防止)
    if not np.isfinite(dse) or dse < 0.0:
        dse = 0.0
    if not np.isfinite(te_sec) or te_sec < 0.0:
        te_sec = 0.0

    return {
        "y": float(y),
        "depth_source": depth_source,
        "V": float(V),
        "Vc": float(Vc),
        "V_Vc": float(V_Vc),
        "Fr": float(Fr),
        "dse": float(dse),
        "te_sec": float(te_sec),
        "Ks": float(Ks),
        "K_theta": float(K_theta),
        "K_sigma": float(K_sigma),
        "regime": regime
    }


def calc_kt_melville_chiew(t_sec: float, te_sec: float, V_eff_Vc: float) -> float:
    """Melville & Chiew (1999) の時間係数 Kt = ds(t)/dse を算出"""
    if t_sec <= 0.0 or te_sec <= 0.0 or V_eff_Vc <= 0.4:
        return 0.0
    if t_sec >= te_sec:
        return 1.0
    term = abs((1.0 / V_eff_Vc) * np.log(t_sec / te_sec))
    kt = float(np.exp(-0.03 * (term ** 1.6)))
    return min(max(kt, 0.0), 1.0)


def calc_equivalent_time_star(Kt_star: float, te_sec: float, V_eff_Vc: float) -> float:
    """非定常計算用：現在の到達率 Kt* に相当する等価経過時間 t* [sec] を逆算"""
    if Kt_star <= 0.0 or te_sec <= 0.0 or V_eff_Vc <= 0.4:
        return 0.0
    if Kt_star >= 1.0:
        return te_sec
    Kt_clipped = min(max(Kt_star, 1e-12), 1.0 - 1e-12)
    exponent = -V_eff_Vc * ((-np.log(Kt_clipped) / 0.03) ** (1.0 / 1.6))
    return float(te_sec * np.exp(exponent))


def evaluate_experiment_fit(t_pred_min: np.ndarray, ds_pred_mm: np.ndarray, exp_df: pd.DataFrame) -> dict:
    """実測データと理論予測値の誤差指標（RMSE, MAE, MAPE, R2）を算出"""
    clean_exp = exp_df.dropna(subset=["time_min", "ds_exp_mm"]).sort_values("time_min")
    if len(clean_exp) == 0:
        return {"valid": False}

    t_exp = clean_exp["time_min"].values.astype(float)
    ds_exp = clean_exp["ds_exp_mm"].values.astype(float)

    valid_mask = (t_exp >= t_pred_min.min()) & (t_exp <= t_pred_min.max())
    if not np.any(valid_mask):
        return {"valid": False, "reason": "実測データの時刻がシミュレーション時間範囲外です。"}

    t_exp = t_exp[valid_mask]
    ds_exp = ds_exp[valid_mask]
    ds_pred_interp = np.interp(t_exp, t_pred_min, ds_pred_mm)

    errors = ds_pred_interp - ds_exp
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    non_zero_mask = ds_exp > 0.0
    mape = float(np.mean(np.abs(errors[non_zero_mask] / ds_exp[non_zero_mask])) * 100.0) if np.any(non_zero_mask) else np.nan

    ss_res = np.sum((ds_exp - ds_pred_interp) ** 2)
    ss_tot = np.sum((ds_exp - np.mean(ds_exp)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0.0 else np.nan

    comp_df = pd.DataFrame({
        "time_min": t_exp,
        "ds_exp_mm": ds_exp,
        "ds_pred_mm": ds_pred_interp,
        "error_mm (pred - exp)": errors,
        "rel_error_%": np.where(ds_exp > 0, (errors / ds_exp) * 100.0, np.nan)
    })

    return {
        "valid": True,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "comp_df": comp_df
    }


# ==========================================
# 2. タイトル & スライド式モード切替スイッチ
# ==========================================
st.title(":material/waves: 橋脚局所洗掘 時間発展予測・実験検証システム")
st.caption("Ver. 2.1 ｜ Melville & Chiew (1999) 時間発展式 ＋ S/M法 & Melville & Coleman (2000) 法 ＋ 実験誤差検証")

with st.container(border=True):
    col_sw_left, col_sw_mid, col_sw_right = st.columns([2, 1, 2])
    with col_sw_mid:
        is_unsteady = st.toggle("モード切替", value=False, label_visibility="collapsed")
    with col_sw_left:
        if not is_unsteady:
            st.markdown("**:material/trending_flat: 定常流モード (Steady)**  `[選択中]`")
        else:
            st.markdown(":material/trending_flat: 定常流モード (Steady)")
    with col_sw_right:
        if is_unsteady:
            st.markdown("**:material/show_chart: 非定常流モード (Unsteady)**  `[選択中]`")
        else:
            st.markdown(":material/show_chart: 非定常流モード (Unsteady)")


# ==========================================
# 3. サイドバー：共通諸元の入力UI
# ==========================================
with st.sidebar:
    st.header(":material/architecture: 1. 橋脚諸元")
    pier_shape = st.selectbox(
        "橋脚の平面形状",
        ["小判型 (Round-nosed)", "円柱 (Circular)", "矩形 (Rectangular)", "任意係数指定 (Custom)"],
        index=0
    )
    custom_Ks = 1.0
    if pier_shape == "任意係数指定 (Custom)":
        custom_Ks = st.number_input("形状係数 Ks [-]", min_value=0.1, max_value=3.0, value=1.0, step=0.05, format="%.2f")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        pier_b = st.number_input("橋脚幅 b [m]", min_value=0.005, max_value=20.0, value=0.05, step=0.01, format="%.3f")
    with col_p2:
        pier_L = st.number_input("橋脚長さ L [m]", min_value=0.005, max_value=100.0, value=0.15, step=0.01, format="%.3f")
    theta_deg = st.number_input("流れに対する迎え角 θ [deg]", min_value=0.0, max_value=90.0, value=0.0, step=1.0)

    st.divider()
    st.header(":material/straighten: 2. 水路・河道諸元")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        chan_B = st.number_input("水路幅 B [m]", min_value=0.05, max_value=500.0, value=0.50, step=0.05, format="%.3f")
    with col_c2:
        chan_L = st.number_input("水路長さ Lch [m]", min_value=0.1, max_value=5000.0, value=10.0, step=0.5, format="%.2f")

    depth_mode = st.radio(
        "水深 y の決定方法（区別必須）",
        ["実測・設定水深 y を直接入力", "勾配 I と粗度 n から等流水深を逆算"],
        index=0
    )
    use_manning = (depth_mode == "勾配 I と粗度 n から等流水深を逆算")

    col_c3, col_c4 = st.columns(2)
    with col_c3:
        slope_I = st.number_input(
            "河床勾配 I [-]", min_value=0.00001, max_value=0.5, value=0.001, step=0.0005, format="%.5f",
            disabled=not use_manning
        )
    with col_c4:
        manning_n = st.number_input(
            "マニング粗度 n", min_value=0.005, max_value=0.100, value=0.015, step=0.001, format="%.3f",
            disabled=not use_manning
        )

    st.divider()
    st.header(":material/layers: 3. 河床材料・水温")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        d50_mm = st.number_input("代表粒径 d50 [mm]", min_value=0.05, max_value=200.0, value=0.80, step=0.10, format="%.2f")
    with col_s2:
        sed_s = st.number_input("砂粒子比重 s [-]", min_value=1.10, max_value=5.00, value=2.65, step=0.05, format="%.2f")
    col_s3, col_s4 = st.columns(2)
    with col_s3:
        sigma_g = st.number_input("幾何標準偏差 σg [-]", min_value=1.00, max_value=5.00, value=1.20, step=0.05, format="%.2f")
    with col_s4:
        temp_c = st.number_input("水温 T [℃] (0〜40℃)", min_value=0.0, max_value=40.0, value=20.0, step=1.0, format="%.1f")

    st.divider()
    st.header(":material/calculate: 4. 平衡洗掘深 dse モデル")
    dse_method = st.selectbox(
        "dse 算定モデル",
        ["Sheppard / Melville (S/M) 法", "Melville & Coleman (2000) 法", "手入力"],
        index=0
    )
    manual_dse_mm = 0.0
    if dse_method == "手入力":
        manual_dse_mm = st.number_input("指定する平衡洗掘深 dse [mm]", min_value=0.0, max_value=5000.0, value=100.0, step=5.0)


# ==========================================
# 4. 実験条件の自動診断アラート & 理論リファレンス
# ==========================================
blockage_ratio = chan_B / pier_b
if blockage_ratio < 8.0:
    st.warning(
        f"**側壁影響（閉塞効果）の警告**：水路幅/橋脚幅比 $B/b = {blockage_ratio:.1f} < 8.0$ です。側壁の影響で洗掘深が過大になる恐れがあります。",
        icon=":material/warning:"
    )
elif blockage_ratio < 10.0:
    st.info(
        f"**側壁影響の注意**：$B/b = {blockage_ratio:.1f}$（推奨目安は $B/b \\ge 8 \\sim 10$）です。",
        icon=":material/info:"
    )

if chan_L / chan_B < 5.0:
    st.info(
        f"**助走区間の注意**：水路長さ/水路幅比 $L_{{ch}}/B = {chan_L/chan_B:.1f}$ と短めです。橋脚上流側で流れが十分に発達しているか確認してください。",
        icon=":material/info:"
    )

with st.expander("理論式の適用条件・係数定義リファレンス（検証用）", icon=":material/menu_book:"):
    st.markdown("""
    * **洗掘発生限界 $V/V_c = 0.4$ の根拠**：Melville & Chiew (1999) および Melville & Coleman (2000) に基づき、橋脚前面の馬蹄形渦による局所掃流力が限界掃流力を超えて洗掘が始まるしきい値として採用しています。
    * **形状係数 $K_s$ の定義**：Melville & Coleman (2000) に準拠し、円柱・小判型（先端半円形）は $K_s = 1.0$、矩形は $K_s = 1.1$ としています（側面メニューから任意値への上書きも可能）。
    * **水の動粘性係数 $\\nu$**：Poiseuilleの経験式（適用範囲：$0 \\sim 40^\\circ\\text{C}$）を使用しています。
    * **水深 $y$ の区別**：実測水深を用いた場合とマニング式による等流水深を用いた場合では接近流条件が異なるため、結果サマリーおよびCSVに算定区分を明記しています。
    * **非定常モデルの前提条件**：等価時間重ね合わせ法は「埋め戻し（堆積）を伴わない洗掘深の単調増加」を仮定しています。減水期の堆積再現には流砂連続式との連成が必要です。
    """)


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
            y_calc = solve_normal_depth_manning(Q_steady_m3s, chan_B, slope_I, manning_n)
            y_steady = y_calc
            st.metric("マニング等流水深 y（自動算定）", f"{y_steady:.4f} m")
    with col_q3:
        t_end_min = st.number_input("通水時間 t_end [min]", min_value=1.0, max_value=60000.0, value=360.0, step=10.0, format="%.1f")

    if Q_steady_lmin <= 0.0 or y_steady <= 0.0:
        st.error("流量 Q および水深 y は正の値を設定してください。", icon=":material/error:")
        st.stop()

    # 共通コアエンジン実行
    res = core_scour_engine(
        Q_m3s=Q_steady_m3s, y_input=y_steady, use_manning=use_manning,
        b=pier_b, L=pier_L, theta_deg=theta_deg, shape=pier_shape, custom_Ks=custom_Ks,
        B=chan_B, I=slope_I, n=manning_n,
        d50_mm=d50_mm, s=sed_s, sigma_g=sigma_g, temp_c=temp_c,
        dse_method=dse_method, manual_dse=manual_dse_mm / 1000.0
    )

    if res["V_Vc"] > 1.0:
        st.error(
            f"**動的洗掘領域 (Live-bed scour: V/Vc = {res['V_Vc']:.2f} > 1.0)**：Melville & Chiew (1999) の時間発展式は静的洗掘 ($V/V_c \\le 1.0$) 向けのため、時間進行は $V/V_c = 1.0$ でクリップして参考値を表示しています。",
            icon=":material/error:"
        )
    elif res["V_Vc"] <= 0.4:
        st.info(
            f"**無洗掘領域 (V/Vc = {res['V_Vc']:.2f} <= 0.4)**：流速が砂の移動限界の40%以下のため、局所洗掘は発生しません。",
            icon=":material/info:"
        )

    # 実験データ比較パネル（定常流）
    with st.expander("実験実測データとの比較・誤差検証（任意）", icon=":material/science:"):
        enable_exp_steady = st.checkbox("実測洗掘深データをプロットして理論式と誤差比較する", value=False)
        exp_df_steady = pd.DataFrame(columns=["time_min", "ds_exp_mm"])
        if enable_exp_steady:
            col_e1, col_e2 = st.columns([1, 2])
            with col_e1:
                exp_csv_s = st.file_uploader("実測CSV読込 (`time_min`, `ds_exp_mm`)", type=["csv"], key="exp_csv_s")
            with col_e2:
                default_exp_s = pd.DataFrame({
                    "time_min": [5.0, 15.0, 30.0, 60.0, 120.0, 240.0, 360.0],
                    "ds_exp_mm": [48.0, 62.0, 71.0, 80.0, 89.0, 96.0, 101.0]
                })
                if exp_csv_s is not None:
                    raw_exp = pd.read_csv(exp_csv_s)
                    if not {"time_min", "ds_exp_mm"}.issubset(raw_exp.columns):
                        st.error("CSVには `time_min` と `ds_exp_mm` の2列が必要です。", icon=":material/error:")
                    else:
                        exp_df_steady = raw_exp[["time_min", "ds_exp_mm"]]
                else:
                    exp_df_steady = st.data_editor(default_exp_s, num_rows="dynamic", use_container_width=True, key="exp_ed_s")

    # 時系列配列の生成（[min] -> 内部計算用 [sec]）
    t_end_sec = t_end_min * 60.0
    t_log = np.geomspace(1.0, max(t_end_sec, 2.0), 150)
    t_lin = np.linspace(0.0, t_end_sec, 150)
    t_sec_arr = np.unique(np.concatenate(([0.0], t_log, t_lin)))
    t_sec_arr = t_sec_arr[t_sec_arr <= t_end_sec]

    V_eff_Vc = min(res["V_Vc"], 1.0)
    Kt_arr = np.array([calc_kt_melville_chiew(t, res["te_sec"], V_eff_Vc) for t in t_sec_arr])
    ds_m_arr = Kt_arr * res["dse"]

    df_steady = pd.DataFrame({
        "time_min": t_sec_arr / 60.0,
        "time_sec": t_sec_arr,
        "depth_source": res["depth_source"],
        "Q_l_min": Q_steady_lmin,
        "Q_m3s": Q_steady_m3s,
        "y_m": res["y"],
        "V_m_s": res["V"],
        "Vc_m_s": res["Vc"],
        "V_Vc": res["V_Vc"],
        "Kt_percent": Kt_arr * 100.0,
        "ds_mm": ds_m_arr * 1000.0,
        "dse_mm": res["dse"] * 1000.0
    })

    # KPI メトリクス表示
    st.divider()
    st.subheader(f":material/analytics: 予測結果サマリー（定常流 ｜ 水深区分：{res['depth_source']}）")
    k1, k2, k3, k4, k5 = st.columns(5)
    final_ds_mm = ds_m_arr[-1] * 1000.0
    final_kt_pct = Kt_arr[-1] * 100.0
    te_min = res["te_sec"] / 60.0
    te_hr = res["te_sec"] / 3600.0

    k1.metric(f"通水 {t_end_min:.0f} min 後洗掘深 ds", f"{final_ds_mm:.1f} mm", f"到達率 {final_kt_pct:.1f} %")
    k2.metric("最終平衡洗掘深 dse", f"{res['dse']*1000.0:.1f} mm", f"ds/b = {res['dse']/pier_b:.2f}")
    k3.metric("平衡到達時間 te", f"{te_min:.0f} min", f"約 {te_hr:.1f} hr")
    k4.metric("流速強度 V / Vc", f"{res['V_Vc']:.3f}", f"V={res['V']:.3f}, Vc={res['Vc']:.3f} m/s")
    k5.metric("無次元水深 y/b ・ Fr", f"y/b = {res['y']/pier_b:.2f}", f"Fr = {res['Fr']:.3f}")

    # グラフ描画（通常軸 / 対数軸 切替 ＋ 実測データ重ね描き）
    axis_type = st.radio("横軸（時間軸）の表示スケール", ["通常軸 (Linear)", "対数軸 (Log - 初期洗掘の確認用)"], horizontal=True)
    x_axis_mode = "log" if "対数" in axis_type else "linear"

    plot_df = df_steady[df_steady["time_min"] > 0] if x_axis_mode == "log" else df_steady

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=plot_df["time_min"], y=plot_df["ds_mm"],
            mode="lines", name="理論予測洗掘深 ds(t) [mm]",
            line=dict(color="#1f77b4", width=3)
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["time_min"], y=plot_df["Kt_percent"],
            mode="lines", name="到達率 Kt [%]",
            line=dict(color="#ff7f0e", width=2, dash="dash")
        ),
        secondary_y=True
    )

    fit_res_s = {"valid": False}
    if enable_exp_steady and len(exp_df_steady.dropna()) > 0:
        clean_exp_s = exp_df_steady.dropna().sort_values("time_min")
        plot_exp_s = clean_exp_s[clean_exp_s["time_min"] > 0] if x_axis_mode == "log" else clean_exp_s
        fig.add_trace(
            go.Scatter(
                x=plot_exp_s["time_min"], y=plot_exp_s["ds_exp_mm"],
                mode="markers", name="実験実測値 ds,exp [mm]",
                marker=dict(color="#d62728", size=9, symbol="circle")
            ),
            secondary_y=False
        )
        fit_res_s = evaluate_experiment_fit(df_steady["time_min"].values, df_steady["ds_mm"].values, clean_exp_s)

    fig.add_hline(
        y=res["dse"] * 1000.0, line_dash="dot", line_color="gray",
        annotation_text=f"平衡洗掘深 dse = {res['dse']*1000.0:.1f} mm",
        secondary_y=False
    )

    fig.update_layout(
        title="洗掘深 ds(t) および 到達率 Kt の時間発展曲線",
        xaxis_title="通水経過時間 t [min]",
        xaxis_type=x_axis_mode,
        hovermode="x unified",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="洗掘深 ds [mm]", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="到達率 Kt [%]", secondary_y=True, range=[0, 105])
    st.plotly_chart(fig, use_container_width=True)

    if enable_exp_steady:
        if fit_res_s["valid"]:
            st.markdown("**:material/fact_check: 理論予測値 vs 実験実測値 誤差評価指標**")
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("RMSE (二乗平均平方根誤差)", f"{fit_res_s['rmse']:.2f} mm")
            e2.metric("MAE (平均絶対誤差)", f"{fit_res_s['mae']:.2f} mm")
            e3.metric("MAPE (平均相対誤差)", f"{fit_res_s['mape']:.1f} %" if not np.isnan(fit_res_s["mape"]) else "—")
            e4.metric("決定係数 R²", f"{fit_res_s['r2']:.4f}" if not np.isnan(fit_res_s["r2"]) else "—")
            with st.expander("実測時刻ごとの予測値・残差一覧", icon=":material/list_alt:"):
                st.dataframe(fit_res_s["comp_df"].round(3), use_container_width=True)
        elif "reason" in fit_res_s:
            st.warning(fit_res_s["reason"], icon=":material/warning:")

    with st.expander("時系列データテーブルの確認・CSVダウンロード", icon=":material/table_chart:"):
        st.dataframe(df_steady.round(4), use_container_width=True)
        csv_bytes = df_steady.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="計算結果CSVをダウンロード (定常流)",
            data=csv_bytes,
            file_name="scour_steady_result.csv",
            mime="text/csv",
            icon=":material/download:"
        )

else:
    # ------------------------------------------
    # 非定常流モード (Unsteady / ハイドログラフ)
    # ------------------------------------------
    st.subheader(":material/show_chart: 非定常流モード：ハイドログラフ（時系列）の設定")
    col_u1, col_u2 = st.columns([1, 2])
    with col_u1:
        dt_sec = st.number_input("計算時間ステップ Δt [sec]", min_value=1.0, max_value=3600.0, value=30.0, step=5.0)
        uploaded_csv = st.file_uploader("CSVから時系列を読み込む（任意）", type=["csv"], key="hydro_csv")
        st.caption("※CSV必須列: `time_min`, `Q_l_min`（実測水深入力時は `y` 列も必須）")

    with col_u2:
        st.markdown("**時系列テーブル（時間：`min`・流量：`l/min` / 直接編集・行追加が可能）**")
        default_hydro = pd.DataFrame({
            "time_min": [0.0, 30.0, 60.0, 90.0, 120.0, 180.0, 240.0],
            "Q_l_min": [480.0, 900.0, 1500.0, 1320.0, 960.0, 600.0, 480.0],
            "y": [0.10, 0.13, 0.17, 0.16, 0.14, 0.11, 0.10]
        })
        if uploaded_csv is not None:
            hydro_input_df = pd.read_csv(uploaded_csv)
        else:
            cols_to_show = ["time_min", "Q_l_min"] if use_manning else ["time_min", "Q_l_min", "y"]
            hydro_input_df = st.data_editor(
                default_hydro[cols_to_show],
                num_rows="dynamic",
                use_container_width=True
            )

    # 厳密な入力データバリデーション
    required_cols = {"time_min", "Q_l_min"} if use_manning else {"time_min", "Q_l_min", "y"}
    if not required_cols.issubset(hydro_input_df.columns):
        st.error(f"入力データに必須列 `{', '.join(required_cols)}` が不足しています。", icon=":material/error:")
        st.stop()

    hydro_df = hydro_input_df.dropna(subset=list(required_cols)).sort_values("time_min").reset_index(drop=True)
    if len(hydro_df) < 2:
        st.error("時系列データは少なくとも2行以上入力してください。", icon=":material/error:")
        st.stop()

    if hydro_df["time_min"].duplicated().any():
        st.error("時系列データに重複した時刻 (`time_min`) が含まれています。重複を解消してください。", icon=":material/error:")
        st.stop()

    if (hydro_df["Q_l_min"] < 0.0).any():
        st.error("流量 `Q_l_min` に負の値が含まれています。0以上の値を入力してください。", icon=":material/error:")
        st.stop()

    if not use_manning and (hydro_df["y"] <= 0.0).any():
        st.error("水深 `y` に 0 以下の値が含まれています。正の値を入力してください。", icon=":material/error:")
        st.stop()

    # 実験データ比較パネル（非定常流）
    with st.expander("実験実測データとの比較・誤差検証（任意）", icon=":material/science:"):
        enable_exp_unsteady = st.checkbox("実測洗掘深データをプロットして理論式と誤差比較する", value=False)
        exp_df_unsteady = pd.DataFrame(columns=["time_min", "ds_exp_mm"])
        if enable_exp_unsteady:
            col_eu1, col_eu2 = st.columns([1, 2])
            with col_eu1:
                exp_csv_u = st.file_uploader("実測CSV読込 (`time_min`, `ds_exp_mm`)", type=["csv"], key="exp_csv_u")
            with col_eu2:
                default_exp_u = pd.DataFrame({
                    "time_min": [30.0, 60.0, 90.0, 120.0, 180.0, 240.0],
                    "ds_exp_mm": [35.0, 68.0, 82.0, 85.0, 84.0, 83.0]
                })
                if exp_csv_u is not None:
                    raw_exp_u = pd.read_csv(exp_csv_u)
                    if not {"time_min", "ds_exp_mm"}.issubset(raw_exp_u.columns):
                        st.error("CSVには `time_min` と `ds_exp_mm` の2列が必要です。", icon=":material/error:")
                    else:
                        exp_df_unsteady = raw_exp_u[["time_min", "ds_exp_mm"]]
                else:
                    exp_df_unsteady = st.data_editor(default_exp_u, num_rows="dynamic", use_container_width=True, key="exp_ed_u")

    # 線形補間（[min] -> [sec], [l/min] -> [m^3/s]）
    t_raw_sec = hydro_df["time_min"].values.astype(float) * 60.0
    Q_raw_lmin = hydro_df["Q_l_min"].values.astype(float)
    Q_raw_m3s = Q_raw_lmin * LMIN_TO_M3S
    y_raw = hydro_df["y"].values.astype(float) if not use_manning else np.full_like(Q_raw_m3s, 0.15)

    t_start = t_raw_sec[0]
    t_end = t_raw_sec[-1]
    if t_end <= t_start:
        st.error("終了時刻は開始時刻より大きく設定してください。", icon=":material/error:")
        st.stop()

    n_steps = int(np.ceil((t_end - t_start) / dt_sec))
    t_grid = np.linspace(t_start, t_end, n_steps + 1)
    Q_grid_lmin = np.interp(t_grid, t_raw_sec, Q_raw_lmin)
    Q_grid_m3s = np.interp(t_grid, t_raw_sec, Q_raw_m3s)
    y_grid = np.interp(t_grid, t_raw_sec, y_raw)

    # ステップごとの等価時間重ね合わせ計算ループ
    records = []
    ds_curr = 0.0
    live_bed_occurred = False

    st0 = core_scour_engine(
        Q_m3s=Q_grid_m3s[0], y_input=y_grid[0], use_manning=use_manning,
        b=pier_b, L=pier_L, theta_deg=theta_deg, shape=pier_shape, custom_Ks=custom_Ks,
        B=chan_B, I=slope_I, n=manning_n,
        d50_mm=d50_mm, s=sed_s, sigma_g=sigma_g, temp_c=temp_c,
        dse_method=dse_method, manual_dse=manual_dse_mm / 1000.0
    )
    records.append({
        "time_min": t_grid[0] / 60.0,
        "time_sec": t_grid[0],
        "depth_source": st0["depth_source"],
        "Q_l_min": Q_grid_lmin[0],
        "Q_m3s": Q_grid_m3s[0],
        "y_m": st0["y"],
        "V_m_s": st0["V"],
        "Vc_m_s": st0["Vc"],
        "V_Vc": st0["V_Vc"],
        "dse_mm": st0["dse"] * 1000.0,
        "te_min": st0["te_sec"] / 60.0,
        "t_star_min": 0.0,
        "ds_mm": 0.0,
        "regime": st0["regime"]
    })

    for i in range(1, len(t_grid)):
        dt_step = t_grid[i] - t_grid[i - 1]
        Q_bar_m3s = 0.5 * (Q_grid_m3s[i - 1] + Q_grid_m3s[i])
        y_bar = 0.5 * (y_grid[i - 1] + y_grid[i])

        sti = core_scour_engine(
            Q_m3s=Q_bar_m3s, y_input=y_bar, use_manning=use_manning,
            b=pier_b, L=pier_L, theta_deg=theta_deg, shape=pier_shape, custom_Ks=custom_Ks,
            B=chan_B, I=slope_I, n=manning_n,
            d50_mm=d50_mm, s=sed_s, sigma_g=sigma_g, temp_c=temp_c,
            dse_method=dse_method, manual_dse=manual_dse_mm / 1000.0
        )

        if sti["V_Vc"] > 1.0:
            live_bed_occurred = True

        dse_i = sti["dse"]
        te_i = sti["te_sec"]
        V_eff_Vc = min(sti["V_Vc"], 1.0)
        t_star_sec = 0.0

        # 3パターンの状態分岐：洗掘進行 vs 深さ維持（単調増加仮定）
        if dse_i > 0.0 and ds_curr < dse_i:
            if ds_curr <= 0.0:
                t_star_sec = 0.0
            else:
                Kt_star = ds_curr / dse_i
                t_star_sec = calc_equivalent_time_star(Kt_star, te_i, V_eff_Vc)

            t_next = t_star_sec + dt_step
            Kt_new = calc_kt_melville_chiew(t_next, te_i, V_eff_Vc)
            ds_curr = max(ds_curr, Kt_new * dse_i)

        records.append({
            "time_min": t_grid[i] / 60.0,
            "time_sec": t_grid[i],
            "depth_source": sti["depth_source"],
            "Q_l_min": Q_grid_lmin[i],
            "Q_m3s": Q_grid_m3s[i],
            "y_m": sti["y"],
            "V_m_s": sti["V"],
            "Vc_m_s": sti["Vc"],
            "V_Vc": sti["V_Vc"],
            "dse_mm": dse_i * 1000.0,
            "te_min": te_i / 60.0,
            "t_star_min": t_star_sec / 60.0,
            "ds_mm": ds_curr * 1000.0,
            "regime": sti["regime"]
        })

    df_unsteady = pd.DataFrame(records)

    if live_bed_occurred:
        st.warning(
            "**ピーク時に動的洗掘領域 (V/Vc > 1.0) へ遷移しました**：当該区間は $V/V_c = 1.0$ でクリップして計算しています。実際の洗掘進行は計算値より速くなる可能性があります。",
            icon=":material/warning:"
        )

    # KPI メトリクス表示
    st.divider()
    st.subheader(f":material/analytics: 予測結果サマリー（非定常流 ｜ 水深区分：{st0['depth_source']}）")
    max_ds_mm = df_unsteady["ds_mm"].max()
    max_ds_time_min = df_unsteady.loc[df_unsteady["ds_mm"].idxmax(), "time_min"]
    peak_dse_mm = df_unsteady["dse_mm"].max()
    peak_Q_lmin = df_unsteady["Q_l_min"].max()
    peak_VVc = df_unsteady["V_Vc"].max()

    uk1, uk2, uk3, uk4 = st.columns(4)
    uk1.metric("最大到達洗掘深 ds,max", f"{max_ds_mm:.1f} mm", f"発生時刻 t = {max_ds_time_min:.1f} min")
    uk2.metric("ピーク時瞬間平衡深 dse,peak", f"{peak_dse_mm:.1f} mm", f"到達比 {max_ds_mm/peak_dse_mm*100 if peak_dse_mm>0 else 0:.1f} %")
    uk3.metric("ピーク流量 Qmax", f"{peak_Q_lmin:.1f} l/min", f"最大 V/Vc = {peak_VVc:.2f}")
    uk4.metric("計算ステップ数", f"{len(df_unsteady)-1} steps", f"Δt = {dt_sec:.0f} sec")

    # 2段連動グラフ
    fig_u = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=("1. ハイドログラフ（流量 Q [l/min] と 流速強度 V/Vc）", "2. 洗掘深の推移（累積洗掘深 ds vs 瞬間平衡洗掘深 dse）"),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    fig_u.add_trace(
        go.Scatter(x=df_unsteady["time_min"], y=df_unsteady["Q_l_min"], name="流量 Q [l/min]", line=dict(color="#1f77b4", width=2.5)),
        row=1, col=1, secondary_y=False
    )
    fig_u.add_trace(
        go.Scatter(x=df_unsteady["time_min"], y=df_unsteady["V_Vc"], name="流速強度 V/Vc [-]", line=dict(color="#2ca02c", width=2, dash="dot")),
        row=1, col=1, secondary_y=True
    )
    fig_u.add_hline(y=0.4, line_dash="dash", line_color="gray", annotation_text="洗掘開始限界 (V/Vc=0.4)", row=1, col=1, secondary_y=True)

    fig_u.add_trace(
        go.Scatter(x=df_unsteady["time_min"], y=df_unsteady["ds_mm"], name="累積洗掘深 ds(t) [mm]", line=dict(color="#d62728", width=3)),
        row=2, col=1
    )
    fig_u.add_trace(
        go.Scatter(x=df_unsteady["time_min"], y=df_unsteady["dse_mm"], name="瞬間平衡洗掘深 dse(t) [mm]", line=dict(color="#7f7f7f", width=2, dash="dash")),
        row=2, col=1
    )

    fit_res_u = {"valid": False}
    if enable_exp_unsteady and len(exp_df_unsteady.dropna()) > 0:
        clean_exp_u = exp_df_unsteady.dropna().sort_values("time_min")
        fig_u.add_trace(
            go.Scatter(
                x=clean_exp_u["time_min"], y=clean_exp_u["ds_exp_mm"],
                mode="markers", name="実験実測値 ds,exp [mm]",
                marker=dict(color="#111111", size=9, symbol="diamond")
            ),
            row=2, col=1
        )
        fit_res_u = evaluate_experiment_fit(df_unsteady["time_min"].values, df_unsteady["ds_mm"].values, clean_exp_u)

    fig_u.update_layout(
        height=640,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1)
    )
    fig_u.update_yaxes(title_text="流量 Q [l/min]", row=1, col=1, secondary_y=False, rangemode="tozero")
    fig_u.update_yaxes(title_text="流速強度 V/Vc [-]", row=1, col=1, secondary_y=True, rangemode="tozero")
    fig_u.update_yaxes(title_text="洗掘深 [mm]", row=2, col=1, rangemode="tozero")
    fig_u.update_xaxes(title_text="経過時間 t [min]", row=2, col=1)

    st.plotly_chart(fig_u, use_container_width=True)

    if enable_exp_unsteady:
        if fit_res_u["valid"]:
            st.markdown("**:material/fact_check: 理論予測値 vs 実験実測値 誤差評価指標**")
            eu1, eu2, eu3, eu4 = st.columns(4)
            eu1.metric("RMSE (二乗平均平方根誤差)", f"{fit_res_u['rmse']:.2f} mm")
            eu2.metric("MAE (平均絶対誤差)", f"{fit_res_u['mae']:.2f} mm")
            eu3.metric("MAPE (平均相対誤差)", f"{fit_res_u['mape']:.1f} %" if not np.isnan(fit_res_u["mape"]) else "—")
            eu4.metric("決定係数 R²", f"{fit_res_u['r2']:.4f}" if not np.isnan(fit_res_u["r2"]) else "—")
            with st.expander("実測時刻ごとの予測値・残差一覧", icon=":material/list_alt:"):
                st.dataframe(fit_res_u["comp_df"].round(3), use_container_width=True)
        elif "reason" in fit_res_u:
            st.warning(fit_res_u["reason"], icon=":material/warning:")

    with st.expander("非定常ステップ計算データテーブルの確認・CSVダウンロード", icon=":material/table_chart:"):
        st.dataframe(df_unsteady.round(4), use_container_width=True)
        csv_bytes_u = df_unsteady.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="計算結果CSVをダウンロード (非定常流)",
            data=csv_bytes_u,
            file_name="scour_unsteady_result.csv",
            mime="text/csv",
            icon=":material/download:"
        )