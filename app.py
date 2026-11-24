import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# 網頁基本設定
st.set_page_config(page_title="盈富基金息率 SD 倉位系統 (滾動5年版)", page_icon="💰", layout="wide")

@st.cache_data(ttl=3600)
def fetch_tracker_data():
    ticker = yf.Ticker("2800.HK")
    raw = ticker.history(start="2005-01-01", auto_adjust=False)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw = raw.dropna(subset=["Close", "Adj Close"])

    # 2800.HK 每年派息 2 次，取最近 2 次派息總和作為過去 12 個月派息 (TTM)
    divs = raw[raw["Dividends"] > 0]["Dividends"]
    ttm_div = divs.rolling(2).sum().reindex(raw.index).ffill()

    df = pd.DataFrame(index=raw.index)
    df["Close"] = raw["Close"]
    df["Adj_Close"] = raw["Adj Close"]
    df["TTM_Div"] = ttm_div
    df["Yield"] = (df["TTM_Div"] / df["Close"]) * 100
    df = df.dropna(subset=["Yield"]).copy()

    # 滾動過去 5 年 (1825 日曆日) 均值與標準差，零上帝視角
    # min_periods=252：首年預熱，第 1 至 5 年只用當時已有嘅過去數據，滿 5 年後嚴格滾動 5 年
    df["Mean_5Y"] = df["Yield"].rolling("1825D", min_periods=252).mean()
    df["Std_5Y"] = df["Yield"].rolling("1825D", min_periods=252).std()
    df["Z"] = (df["Yield"] - df["Mean_5Y"]) / df["Std_5Y"]

    df = df.dropna(subset=["Z"]).copy()
    return df, divs

def get_tier_info(z):
    if z >= 2.0:
        return 1.0, "🟢 極度便宜 (+2SD 以上)", "100% 滿倉 (5/5 倉)", "success", "息率高於近5年均值逾兩個標準差，估值極度便宜，建議滿倉配置。"
    elif z >= 1.0:
        return 0.8, "🟢 偏便宜 (+1SD 至 +2SD)", "80% 倉位 (4/5 倉)", "success", "息率具明顯吸引力，屬於價值低吸區，可維持高倉位收息兼等反彈。"
    elif z >= 0.0:
        return 0.6, "🟡 略便宜 (均值至 +1SD)", "60% 倉位 (3/5 倉)", "info", "估值略低於近5年平均，適合維持中高倉位穩健持貨。"
    elif z >= -1.0:
        return 0.4, "🟡 略貴 (-1SD 至均值)", "40% 倉位 (2/5 倉)", "warning", "估值略高於近5年平均，建議偏防守配置，保留一半以上現金。"
    elif z >= -2.0:
        return 0.2, "🔴 偏貴 (-2SD 至 -1SD)", "20% 倉位 (1/5 倉)", "error", "息率跌至近5年偏低水平，市場情緒偏熱，建議分批獲利減至輕倉。"
    else:
        return 0.0, "🔴 極度昂貴 (-2SD 以下)", "0% 清倉 (0/5 倉)", "error", "息率處於近5年極低泡沫區，建議全面獲利離場，現金為王等待回調。"

# 標題與手動更新掣
col_title, col_btn = st.columns([4, 1])
with col_title:
    st.title("💰 盈富基金 (2800.HK) 息率 SD 系統")
    hk_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    st.caption(f"香港時間更新：{hk_time}｜採用「滾動過去 5 年均值」（零上帝視角）")
with col_btn:
    st.write("")
    if st.button("🔄 立即更新數據", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with st.spinner("正在從 Yahoo Finance 下載盈富基金歷史價格與派息數據..."):
    df, divs = fetch_tracker_data()

# 最新一日數據
latest = df.iloc[-1]
latest_date = df.index[-1].strftime("%Y-%m-%d")
latest_price = latest["Close"]
latest_div = latest["TTM_Div"]
latest_yield = latest["Yield"]
mean_5y = latest["Mean_5Y"]
std_5y = latest["Std_5Y"]
latest_z = latest["Z"]

target_pos, zone_label, pos_text, alert_type, advice = get_tier_info(latest_z)

tab1, tab2 = st.tabs(["🎯 今日息率與建議倉位", "📊 零上帝視角回測"])

# ================= 分頁 1：今日息率與建議倉位 =================
with tab1:
    banner_msg = (
        f"### 今日建議倉位：{pos_text} — {zone_label}\n"
        f"{advice}（最後交易日：{latest_date}）"
    )
    if alert_type == "success":
        st.success(banner_msg)
    elif alert_type == "info":
        st.info(banner_msg)
    elif alert_type == "warning":
        st.warning(banner_msg)
    else:
        st.error(banner_msg)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("2800.HK 最新收市價", f"${latest_price:.2f}")
    c2.metric("最新股息率 (TTM)", f"{latest_yield:.2f}%", f"近5年均值: {mean_5y:.2f}%", delta_color="off")
    c3.metric("標準差位置 (Z-Score)", f"{latest_z:+.2f} SD", "正數=偏平｜負數=偏貴", delta_color="off")
    c4.metric("過去12個月總派息", f"${latest_div:.2f}", "最近2次派息合計", delta_color="off")

    st.divider()

    st.subheader("📋 6 級息率階梯與 2800.HK 對應參考價（按近 5 年均值）")
    st.caption(f"按今日近 5 年息率均值 ({mean_5y:.2f}%)、標準差 ({std_5y:.2f}%) 及最新 TTM 派息 (${latest_div:.2f}) 自動換算。")

    y_p2 = mean_5y + 2 * std_5y
    y_p1 = mean_5y + std_5y
    y_m1 = mean_5y - std_5y
    y_m2 = mean_5y - 2 * std_5y

    p_p2 = latest_div / (y_p2 / 100)
    p_p1 = latest_div / (y_p1 / 100)
    p_mean = latest_div / (mean_5y / 100)
    p_m1 = latest_div / (y_m1 / 100)
    p_m2 = latest_div / (y_m2 / 100)

    ladder_df = pd.DataFrame([
        {"估值區間": "🟢 極度便宜 (> +2 SD)", "建議倉位": "100% (5/5 倉)", "觸發息率": f"≥ {y_p2:.2f}%", "2800.HK 參考價": f"${p_p2:.2f} 或以下", "目前位置": "👈 今日位置" if latest_z >= 2.0 else ""},
        {"估值區間": "🟢 偏便宜 (+1 至 +2 SD)", "建議倉位": "80% (4/5 倉)", "觸發息率": f"{y_p1:.2f}% – {y_p2:.2f}%", "2800.HK 參考價": f"${p_p2:.2f} – ${p_p1:.2f}", "目前位置": "👈 今日位置" if 1.0 <= latest_z < 2.0 else ""},
        {"估值區間": "🟡 略便宜 (均值至 +1 SD)", "建議倉位": "60% (3/5 倉)", "觸發息率": f"{mean_5y:.2f}% – {y_p1:.2f}%", "2800.HK 參考價": f"${p_p1:.2f} – ${p_mean:.2f}", "目前位置": "👈 今日位置" if 0.0 <= latest_z < 1.0 else ""},
        {"估值區間": "🟡 略貴 (-1 SD 至均值)", "建議倉位": "40% (2/5 倉)", "觸發息率": f"{y_m1:.2f}% – {mean_5y:.2f}%", "2800.HK 參考價": f"${p_mean:.2f} – ${p_m1:.2f}", "目前位置": "👈 今日位置" if -1.0 <= latest_z < 0.0 else ""},
        {"估值區間": "🔴 偏貴 (-2 至 -1 SD)", "建議倉位": "20% (1/5 倉)", "觸發息率": f"{y_m2:.2f}% – {y_m1:.2f}%", "2800.HK 參考價": f"${p_m1:.2f} – ${p_m2:.2f}", "目前位置": "👈 今日位置" if -2.0 <= latest_z < -1.0 else ""},
        {"估值區間": "🔴 極度昂貴 (< -2 SD)", "建議倉位": "0% (0/5 倉)", "觸發息率": f"< {y_m2:.2f}%", "2800.HK 參考價": f"${p_m2:.2f} 或以上", "目前位置": "👈 今日位置" if latest_z < -2.0 else ""},
    ])
    st.dataframe(ladder_df, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("📈 盈富基金股息率與「動態 5 年 SD 通道」走勢 (%)")
    chart_df = pd.DataFrame(index=df.index)
    chart_df["2800.HK 股息率 (%)"] = df["Yield"]
    chart_df["+2SD (100%滿倉線)"] = df["Mean_5Y"] + 2 * df["Std_5Y"]
    chart_df["+1SD (80%線)"] = df["Mean_5Y"] + df["Std_5Y"]
    chart_df["滾動5年均值"] = df["Mean_5Y"]
    chart_df["-1SD (20%線)"] = df["Mean_5Y"] - df["Std_5Y"]
    chart_df["-2SD (0%清倉線)"] = df["Mean_5Y"] - 2 * df["Std_5Y"]
    st.line_chart(chart_df)

    with st.expander("💵 查看盈富基金 (2800.HK) 最近 6 次派息紀錄"):
        recent_divs = divs.tail(6).sort_index(ascending=False).reset_index()
        recent_divs.columns = ["除淨日", "每股派息 (港元)"]
        recent_divs["除淨日"] = recent_divs["除淨日"].dt.strftime("%Y-%m-%d")
        st.dataframe(recent_divs, use_container_width=True, hide_index=True)

# ================= 分頁 2：零上帝視角回測 =================
with tab2:
    st.subheader("📊 滾動 5 年息率 SD 策略：零上帝視角回測（含股息再投資）")
    st.caption("每日只使用「當時已發生嘅過去 5 年數據」計算倉位，完全排除未來函數。")

    opt1, opt2 = st.columns(2)
    with opt1:
        rebal_freq = st.radio(
            "調倉頻率設定",
            ["每月月底調倉一次（推薦：大幅慳手續費）", "每日收市即時調倉"],
            horizontal=False
        )
    with opt2:
        cash_rate = st.slider(
            "空倉現金年息假設（例如擺定期／貨幣基金 %）",
            min_value=0.0, max_value=5.0, value=2.5, step=0.5
        )

    bt = df.copy()
    conditions = [
        bt["Z"] >= 2.0,
        (bt["Z"] >= 1.0) & (bt["Z"] < 2.0),
        (bt["Z"] >= 0.0) & (bt["Z"] < 1.0),
        (bt["Z"] >= -1.0) & (bt["Z"] < 0.0),
        (bt["Z"] >= -2.0) & (bt["Z"] < -1.0),
        bt["Z"] < -2.0
    ]
    choices = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    bt["Daily_Target"] = np.select(conditions, choices, default=0.6)

    if "每月月底" in rebal_freq:
        # 版本通用寫法：找出每月最後一個交易日更新目標倉位
        ym = bt.index.to_period("M")
        is_month_end = ym != pd.Series(ym, index=bt.index).shift(-1)
        bt["Target_Pos"] = bt["Daily_Target"].where(is_month_end).ffill().bfill()
    else:
        bt["Target_Pos"] = bt["Daily_Target"]

    bt["Position"] = bt["Target_Pos"].shift(1).fillna(0.6)
    trades = int((bt["Position"].diff().abs() > 0).sum())

    bt["Ret_1d"] = bt["Adj_Close"].pct_change().fillna(0)
    daily_cash_ret = (1 + cash_rate / 100) ** (1 / 252) - 1
    bt["Strat_Ret"] = bt["Ret_1d"] * bt["Position"] + daily_cash_ret * (1 - bt["Position"])

    bt["長揸盈富基金 (含息)"] = (1 + bt["Ret_1d"]).cumprod() * 100
    bt["滾動5年息率策略"] = (1 + bt["Strat_Ret"]).cumprod() * 100

    years = (bt.index[-1] - bt.index[0]).days / 365.25

    def calc_metrics(eq, ret):
        tot = (eq.iloc[-1] / 100 - 1) * 100
        cagr = ((eq.iloc[-1] / 100) ** (1 / years) - 1) * 100
        mdd = ((eq / eq.cummax()) - 1).min() * 100
        sh = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else 0
        return tot, cagr, mdd, sh

    bh_t, bh_c, bh_d, bh_s = calc_metrics(bt["長揸盈富基金 (含息)"], bt["Ret_1d"])
    st_t, st_c, st_d, st_s = calc_metrics(bt["滾動5年息率策略"], bt["Strat_Ret"])
    avg_pos = bt["Position"].mean() * 100

    st.caption(
        f"回測區間：{bt.index[0].strftime('%Y-%m-%d')} 至 {latest_date}（約 {years:.1f} 年）｜"
        f"總調倉次數：{trades} 次（平均每年 {trades/years:.1f} 次）｜平均持股倉位：{avg_pos:.1f}%"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("策略總回報", f"{st_t:+.1f}%", f"年化 {st_c:+.2f}%")
    m2.metric("長揸 2800.HK 總回報", f"{bh_t:+.1f}%", f"年化 {bh_c:+.2f}%")
    m3.metric("策略最大回撤", f"{st_d:.1f}%", f"夏普 {st_s:.2f}", delta_color="off")
    m4.metric("長揸最大回撤", f"{bh_d:.1f}%", f"夏普 {bh_s:.2f}", delta_color="off")

    st.line_chart(bt[["長揸盈富基金 (含息)", "滾動5年息率策略"]])
    st.caption("註：已計入盈富基金股息再投資；未扣除券商交易平台費及買賣差價（2800.HK 免印花稅）。")
