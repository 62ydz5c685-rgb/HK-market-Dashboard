import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# 網頁基本設定
st.set_page_config(page_title="港股逆向價值波段系統", page_icon="🎯", layout="wide")

HSI_STOCKS = [
    "0700.HK", "9988.HK", "0005.HK", "1299.HK", "3690.HK",
    "0939.HK", "0941.HK", "1398.HK", "0388.HK", "2318.HK",
    "0883.HK", "1810.HK", "1211.HK", "0981.HK", "3988.HK",
    "0011.HK", "0001.HK", "0002.HK", "0003.HK", "0006.HK",
    "0016.HK", "0027.HK", "0066.HK", "0175.HK", "0267.HK",
    "0288.HK", "0291.HK", "0386.HK", "0669.HK", "0688.HK",
    "0762.HK", "0823.HK", "0857.HK", "0960.HK", "0968.HK",
    "0992.HK", "1038.HK", "1093.HK", "1109.HK", "1113.HK",
    "1177.HK", "1928.HK", "2020.HK", "2269.HK", "2313.HK",
    "2319.HK", "2382.HK", "2628.HK", "9618.HK", "9999.HK"
]

@st.cache_data(ttl=3600)
def fetch_market_data():
    hsi = yf.Ticker("^HSI").history(period="2y")
    tracker = yf.Ticker("2800.HK").history(period="6mo")
    breadth_data = yf.download(
        HSI_STOCKS, period="6mo", group_by="ticker", progress=False, threads=True
    )
    return hsi, tracker, breadth_data

@st.cache_data(ttl=86400)
def run_backtest_data():
    hsi = yf.Ticker("^HSI").history(period="4y")
    stocks_close = yf.download(HSI_STOCKS, period="4y", progress=False, threads=True)["Close"]

    for d in [hsi, stocks_close]:
        d.index = pd.to_datetime(d.index).tz_localize(None)

    df = pd.DataFrame(index=hsi.index)
    df["HSI"] = hsi["Close"]
    df["MA20"] = df["HSI"].rolling(20).mean()
    df["MA50"] = df["HSI"].rolling(50).mean()
    
    # 恒指與20天線乖離率 (BIAS)
    df["Bias_20"] = (df["HSI"] - df["MA20"]) / df["MA20"] * 100

    # 藍籌市寬計算
    stocks_aligned = stocks_close.reindex(df.index).ffill()
    stocks_ma50 = stocks_aligned.rolling(50).mean()
    valid_cnt = stocks_ma50.notna().sum(axis=1)
    above_cnt = (stocks_aligned > stocks_ma50).sum(axis=1)
    df["Breadth"] = (above_cnt / valid_cnt * 100).fillna(0)

    # 逆向波段評分邏輯 (Mean Reversion)
    # 1. 市寬逆向評分: <25%超賣(2分), 25%-60%中性(1分), >60%過熱(0分)
    df["S_Breadth"] = np.where(df["Breadth"] < 25, 2, np.where(df["Breadth"] <= 60, 1, 0))

    # 2. 乖離率逆向評分: <-4%恐慌低吸(2分), -4%至+3%正常(1分), >+3%追高過熱(0分)
    df["S_Bias"] = np.where(df["Bias_20"] < -4.0, 2, np.where(df["Bias_20"] <= 3.0, 1, 0))

    # 總評分 (滿分 4 分)
    df["Score"] = df["S_Breadth"] + df["S_Bias"]
    df = df.dropna(subset=["MA50"]).copy()

    df["Zone"] = np.where(
        df["Score"] >= 3, "🟢 恐慌低吸區 (3-4分)",
        np.where(df["Score"] >= 2, "🟡 中性持有區 (2分)", "🔴 過熱獲利區 (0-1分)")
    )

    df["Ret_1d"] = df["HSI"].pct_change().fillna(0)
    df["Fwd_20d"] = df["HSI"].shift(-20) / df["HSI"] - 1
    df["Fwd_60d"] = df["HSI"].shift(-60) / df["HSI"] - 1

    # 模擬策略倉位：恐慌超賣時 100% 滿倉抄底；中性時 50% 持股；過熱時 0% 獲利離場避險
    pos = np.where(df["Score"] >= 3, 1.0, np.where(df["Score"] >= 2, 0.5, 0.0))
    df["Position"] = pd.Series(pos, index=df.index).shift(1).fillna(0)
    df["Strat_Ret"] = df["Ret_1d"] * df["Position"]

    df["恒指長揸 (Buy & Hold)"] = (1 + df["Ret_1d"]).cumprod() * 100
    df["逆向波段策略"] = (1 + df["Strat_Ret"]).cumprod() * 100

    return df

# 標題與更新按鈕
col_title, col_btn = st.columns([4, 1])
with col_title:
    st.title("🎯 港股逆向價值波段系統")
    hk_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    st.caption(f"香港時間更新：{hk_time}（專為港股上落市設計：逢恐慌超賣分注吸納，過熱逢高分批獲利）")
with col_btn:
    st.write("")
    if st.button("🔄 立即更新數據", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

tab1, tab2 = st.tabs(["🎯 今日波段體檢", "📊 逆向策略回測 (Backtest)"])

# ================= 分頁 1：今日波段體檢 =================
with tab1:
    with st.spinner("正在下載港股最新收市數據..."):
        hsi, tracker, breadth_data = fetch_market_data()

    hsi["MA20"] = hsi["Close"].rolling(20).mean()
    hsi["MA50"] = hsi["Close"].rolling(50).mean()
    hsi_close = hsi["Close"].iloc[-1]
    hsi_ma20 = hsi["MA20"].iloc[-1]
    hsi_ma50 = hsi["MA50"].iloc[-1]
    bias_20 = (hsi_close - hsi_ma20) / hsi_ma20 * 100
    hsi_date = hsi.index[-1].strftime("%Y-%m-%d")

    # 藍籌市寬
    above_50, valid_stocks = 0, 0
    for ticker in HSI_STOCKS:
        try:
            closes = breadth_data[ticker]["Close"].dropna()
            if len(closes) >= 50:
                valid_stocks += 1
                if closes.iloc[-1] > closes.rolling(50).mean().iloc[-1]:
                    above_50 += 1
        except Exception:
            continue

    breadth_pct = (above_50 / valid_stocks * 100) if valid_stocks > 0 else 0

    # 逆向指標計分
    if breadth_pct < 25:
        score_breadth, light_breadth = 2, "🟢 極度超賣 (2/2分)"
        desc_breadth = "市寬跌破 25%，大市極度恐慌，具備高勝率反彈條件"
    elif breadth_pct <= 60:
        score_breadth, light_breadth = 1, "🟡 正常區間 (1/2分)"
        desc_breadth = "市寬處於 25%-60%，健康整固中"
    else:
        score_breadth, light_breadth = 0, "🔴 嚴重過熱 (0/2分)"
        desc_breadth = "市寬超過 60%，市場追高情緒過熱，隨時見頂回吐"

    if bias_20 < -4.0:
        score_bias, light_bias = 2, "🟢 深度負乖離 (2/2分)"
        desc_bias = f"低於20天線 {abs(bias_20):.1f}%，超賣嚴重，跌無可跌"
    elif bias_20 <= 3.0:
        score_bias, light_bias = 1, "🟡 正常波動 (1/2分)"
        desc_bias = f"偏離20天線 {bias_20:+.1f}%，走勢平穩"
    else:
        score_bias, light_bias = 0, "🔴 正乖離過高 (0/2分)"
        desc_bias = f"高於20天線 {bias_20:+.1f}%，短線急抽過度，忌追高"

    total_score = score_breadth + score_bias

    st.divider()
    if total_score >= 3:
        st.success(f"### 總評分：{total_score} / 4 分 — 🟢 恐慌低吸區 (建議 100% 分注建倉)\n市場進入極端超賣或深度負乖離，港股勝率極高，適合分注吸納盈富基金或優質龍頭股。（恒指交易日：{hsi_date}）")
    elif total_score == 2:
        st.info(f"### 總評分：{total_score} / 4 分 — 🟡 中性持有區 (建議 50% 倉位)\n市況處於正常波動區間，已持貨者可安心讓利潤奔跑，未持貨者不宜追高。（恒指交易日：{hsi_date}）")
    else:
        st.error(f"### 總評分：{total_score} / 4 分 — 🔴 過熱獲利區 (建議 0%-20% 空倉/避險)\n技術面出現嚴重超買或正乖離過大，歷史上容易出現深度回調，建議果斷逢高分批止賺鎖定利潤。（恒指交易日：{hsi_date}）")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1. 藍籌市寬狀態")
        st.metric("企穩 50 天線藍籌比例", f"{breadth_pct:.1f}%", f"{above_50} / {valid_stocks} 隻")
        st.markdown(f"**狀態：{light_breadth}**")
        st.caption(desc_breadth)
    with c2:
        st.subheader("2. 恒指 20 天線乖離率 (BIAS)")
        st.metric("恒指收市價與乖離", f"{hsi_close:,.0f}", f"{bias_20:+.2f}% 偏離 20MA")
        st.markdown(f"**狀態：{light_bias}**")
        st.caption(desc_bias)

    st.divider()
    st.subheader("恒生指數走勢與 20 天、50 天線")
    st.line_chart(hsi[["Close", "MA20", "MA50"]].tail(250).rename(columns={"Close": "恒指收市", "MA20": "20天線", "MA50": "50天線"}))

# ================= 分頁 2：歷史回測 =================
with tab2:
    st.subheader("📊 逆向價值波段策略：歷史回測驗證")
    st.caption("回測邏輯：當市場恐慌超賣 (3-4分) 時 100% 滿倉抄底；中性 (2分) 時 50% 持貨；過熱 (0-1分) 時 0% 空倉獲利離場。")

    if st.button("🚀 開始執行逆向策略回測", type="primary"):
        with st.spinner("正在計算過去 4 年每日逆向評分與收益率，約需 10 秒..."):
            bt = run_backtest_data()

        start_d = bt.index[0].strftime("%Y-%m-%d")
        end_d = bt.index[-1].strftime("%Y-%m-%d")
        st.success(f"回測完成！區間：{start_d} 至 {end_d}（共 {len(bt)} 個交易日）")

        st.markdown("#### 1. 各評分區間下，恒指之後的表現統計")
        summary_rows = []
        for zone in ["🟢 恐慌低吸區 (3-4分)", "🟡 中性持有區 (2分)", "🔴 過熱獲利區 (0-1分)"]:
            sub = bt[bt["Zone"] == zone]
            if len(sub) > 0:
                avg_20 = sub["Fwd_20d"].mean() * 100
                win_20 = (sub["Fwd_20d"] > 0).mean() * 100
                avg_60 = sub["Fwd_60d"].mean() * 100
                win_60 = (sub["Fwd_60d"] > 0).mean() * 100
                summary_rows.append({
                    "狀態分區": zone,
                    "出現日數": f"{len(sub)} 日",
                    "後20日平均升跌": f"{avg_20:+.2f}%",
                    "後20日勝率": f"{win_20:.1f}%",
                    "後60日平均升跌": f"{avg_60:+.2f}%",
                    "後60日勝率": f"{win_60:.1f}%",
                })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        st.markdown("#### 2. 逆向波段策略 vs 長揸恒指（起點 = 100）")
        bh_ret = bt["恒指長揸 (Buy & Hold)"].iloc[-1] - 100
        st_ret = bt["逆向波段策略"].iloc[-1] - 100

        def max_dd(series):
            return ((series / series.cummax()) - 1).min() * 100

        bh_dd = max_dd(bt["恒指長揸 (Buy & Hold)"])
        st_dd = max_dd(bt["逆向波段策略"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("逆向策略總回報", f"{st_ret:+.1f}%")
        m2.metric("長揸恒指總回報", f"{bh_ret:+.1f}%")
        m3.metric("逆向策略最大回撤", f"{st_dd:.1f}%")
        m4.metric("長揸恒指最大回撤", f"{bh_dd:.1f}%")

        st.line_chart(bt[["恒指長揸 (Buy & Hold)", "逆向波段策略"]])
