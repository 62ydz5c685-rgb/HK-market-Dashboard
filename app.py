import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# 網頁基本設定
st.set_page_config(page_title="港股大市紅綠燈 (加速改良版)", page_icon="🚦", layout="wide")

# 50隻主要恒指藍籌股（用於計算港股市寬）
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
    # 抓取過去 4 年數據（預熱 50 天線後實測約 3.5 年）
    hsi = yf.Ticker("^HSI").history(period="4y")
    tracker = yf.Ticker("2800.HK").history(period="4y")
    stocks_close = yf.download(HSI_STOCKS, period="4y", progress=False, threads=True)["Close"]

    for d in [hsi, tracker, stocks_close]:
        d.index = pd.to_datetime(d.index).tz_localize(None)

    df = pd.DataFrame(index=hsi.index)
    df["HSI"] = hsi["Close"]
    df["MA20"] = df["HSI"].rolling(20).mean()
    df["MA50"] = df["HSI"].rolling(50).mean()

    # 1. 敏捷趨勢評分 (20MA & 50MA)
    df["S_Trend"] = np.where(
        (df["HSI"] > df["MA20"]) & (df["MA20"] > df["MA50"]), 2,
        np.where((df["HSI"] < df["MA50"]) & (df["MA20"] < df["MA50"]), 0, 1)
    )

    # 2. 藍籌市寬評分
    stocks_aligned = stocks_close.reindex(df.index).ffill()
    stocks_ma50 = stocks_aligned.rolling(50).mean()
    valid_cnt = stocks_ma50.notna().sum(axis=1)
    above_cnt = (stocks_aligned > stocks_ma50).sum(axis=1)
    df["Breadth"] = (above_cnt / valid_cnt * 100).fillna(0)
    df["S_Breadth"] = np.where(df["Breadth"] > 50, 2, np.where(df["Breadth"] >= 30, 1, 0))

    # 3. 港股機構派發日 (2800.HK 跌幅 >= 0.6% 且放量)
    tr = tracker.reindex(df.index).ffill()
    dist_day = (tr["Close"].pct_change() <= -0.006) & (tr["Volume"] > tr["Volume"].shift(1))
    df["Dist_Days"] = dist_day.rolling(20).sum()
    df["S_Inst"] = np.where(df["Dist_Days"] <= 1, 2, np.where(df["Dist_Days"] <= 3, 1, 0))

    # 總評分 (滿分 6 分)
    df["Score"] = df["S_Trend"] + df["S_Breadth"] + df["S_Inst"]
    df = df.dropna(subset=["MA50"]).copy()

    df["Zone"] = np.where(
        df["Score"] >= 5, "🟢 安全區 (5-6分)",
        np.where(df["Score"] >= 3, "🟡 警戒區 (3-4分)", "🔴 危險區 (0-2分)")
    )

    df["Ret_1d"] = df["HSI"].pct_change().fillna(0)
    df["Fwd_20d"] = df["HSI"].shift(-20) / df["HSI"] - 1
    df["Fwd_60d"] = df["HSI"].shift(-60) / df["HSI"] - 1

    # 模擬策略倉位：5-6分滿倉(1.0)、3-4分半倉(0.5)、0-2分空倉(0.0)
    pos = np.where(df["Score"] >= 5, 1.0, np.where(df["Score"] >= 3, 0.5, 0.0))
    df["Position"] = pd.Series(pos, index=df.index).shift(1).fillna(0)
    df["Strat_Ret"] = df["Ret_1d"] * df["Position"]

    df["恒指長揸 (Buy & Hold)"] = (1 + df["Ret_1d"]).cumprod() * 100
    df["紅綠燈動態倉位策略"] = (1 + df["Strat_Ret"]).cumprod() * 100

    return df

def detect_ftd(df):
    recent = df.tail(60).copy()
    recent["Pct_Change"] = recent["Close"].pct_change()
    recent["Vol_Up"] = recent["Volume"] > recent["Volume"].shift(1)
    recent["Upper_Half"] = recent["Close"] >= (recent["High"] + recent["Low"]) / 2

    lookback = recent.tail(40)
    min_idx = lookback["Low"].idxmin()
    sub = recent.loc[min_idx:].copy()

    rally_day = 0
    rally_low = sub["Low"].iloc[0]
    ftd_info = None

    for i in range(len(sub)):
        row = sub.iloc[i]
        date_str = sub.index[i].strftime("%Y-%m-%d")
        if rally_day == 0:
            if row["Pct_Change"] > 0 or row["Upper_Half"]:
                rally_day = 1
                rally_low = min(rally_low, row["Low"])
        else:
            if row["Low"] < rally_low:
                rally_day = 0
                rally_low = row["Low"]
                ftd_info = None
                if row["Pct_Change"] > 0 or row["Upper_Half"]:
                    rally_day = 1
            else:
                rally_day += 1
                if rally_day >= 4 and ftd_info is None:
                    if row["Pct_Change"] >= 0.0125 and row["Vol_Up"]:
                        ftd_info = {"date": date_str, "day": rally_day, "gain": row["Pct_Change"] * 100}
    return rally_day, rally_low, ftd_info

# 標題與更新掣
col_title, col_btn = st.columns([4, 1])
with col_title:
    st.title("🚦 港股大市紅綠燈 (加速改良版)")
    hk_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    st.caption(f"香港時間更新：{hk_time}（建議每日 16:30 收市後檢視）")
with col_btn:
    st.write("")
    if st.button("🔄 立即更新數據", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

tab1, tab2 = st.tabs(["🚦 今日大市體檢", "📊 歷史回測 (Backtest)"])

# ================= 分頁 1：今日大市體檢 =================
with tab1:
    with st.spinner("正在下載港股最新收市數據..."):
        hsi, tracker, breadth_data = fetch_market_data()

    hsi["MA20"] = hsi["Close"].rolling(20).mean()
    hsi["MA50"] = hsi["Close"].rolling(50).mean()
    hsi_close = hsi["Close"].iloc[-1]
    hsi_ma20 = hsi["MA20"].iloc[-1]
    hsi_ma50 = hsi["MA50"].iloc[-1]
    hsi_date = hsi.index[-1].strftime("%Y-%m-%d")

    # 趨勢判斷
    if hsi_close > hsi_ma20 and hsi_ma20 > hsi_ma50:
        score_trend, light_trend, desc_trend = 2, "🟢 安全", "收市企穩20天線，且20天線高於50天線（強勢）"
    elif hsi_close < hsi_ma50 and hsi_ma20 < hsi_ma50:
        score_trend, light_trend, desc_trend = 0, "🔴 危險", "收市跌穿50天線，且20天線低於50天線（弱勢）"
    else:
        score_trend, light_trend, desc_trend = 1, "🟡 警戒", "處於20天與50天線整固區，方向待定"

    # 市寬計算
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
    if breadth_pct > 50:
        score_breadth, light_breadth = 2, "🟢 安全"
    elif breadth_pct >= 30:
        score_breadth, light_breadth = 1, "🟡 警戒"
    else:
        score_breadth, light_breadth = 0, "🔴 危險"

    # 機構派發日 (2800.HK, 跌幅 >= 0.6% 且成交增加)
    tracker["Pct_Change"] = tracker["Close"].pct_change()
    tracker["Vol_Up"] = tracker["Volume"] > tracker["Volume"].shift(1)
    tracker["Dist_Day"] = (tracker["Pct_Change"] <= -0.006) & tracker["Vol_Up"]
    dist_days = int(tracker.tail(20)["Dist_Day"].sum())

    if dist_days <= 1:
        score_inst, light_inst, desc_inst = 2, "🟢 安全", "近20日大戶未見密集出貨"
    elif dist_days <= 3:
        score_inst, light_inst, desc_inst = 1, "🟡 警戒", "大戶開始逢高減磅"
    else:
        score_inst, light_inst, desc_inst = 0, "🔴 危險", "20日內出現4日或以上放量急跌，機構撤退"

    rally_day, rally_low, ftd_info = detect_ftd(tracker)
    total_score = score_trend + score_breadth + score_inst

    # 總評分卡片 (滿分 6 分)
    if total_score >= 5:
        st.success(f"### 總評分：{total_score} / 6 分 — 🟢 安全持貨區 (建議 100% 倉位)\n大市趨勢明確，市寬健康，大戶未見出貨，可積極操作。（恒指最後交易日：{hsi_date}）")
    elif total_score >= 3:
        st.warning(f"### 總評分：{total_score} / 6 分 — 🟡 提高警覺區 (建議 50% 倉位)\n大市進入整固或分歧階段。建議收緊止蝕，停止大舉開新倉。（恒指最後交易日：{hsi_date}）")
    else:
        st.error(f"### 總評分：{total_score} / 6 分 — 🔴 高度危險區 (建議 0% 空倉/避險)\n技術形態轉弱或出現大戶放量撤退，建議現金為王，避免撈底。（恒指最後交易日：{hsi_date}）")

    # 跟進日狀態
    if ftd_info:
        st.success(f"🚀 **O'Neil 跟進日 (FTD) 已確認！**｜2800.HK 於 **{ftd_info['date']}** 放量急升 **+{ftd_info['gain']:.2f}%**，防守低位 **{rally_low:.2f}**，可留意新升浪機會。")
    elif rally_day >= 4:
        st.warning(f"⏳ **O'Neil 跟進日監察中（嘗試反彈第 {rally_day} 日）**｜2800.HK 暫守住反彈低位 **{rally_low:.2f}**，等待單日放量升 ≥1.25% 確認訊號。")
    elif rally_day >= 1:
        st.info(f"👀 **O'Neil 嘗試反彈初期（第 {rally_day} 日）**｜2800.HK 於 **{rally_low:.2f}** 止跌，需待第 4 日或之後方可確認跟進日。")
    else:
        st.info(f"⚪ **O'Neil 跟進日狀態：未見反彈訊號**｜2800.HK 剛創低位（{rally_low:.2f}），建議耐心觀望。")

    if breadth_pct < 15:
        st.info("🟣 **極度超賣提示**：藍籌市寬低於 15%，隨時出現技術反彈，切忌在低位恐慌斬倉。")

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("1. 敏捷趨勢 (20/50天線)")
        st.metric("恒生指數 (^HSI)", f"{hsi_close:,.0f}", f"20天線: {hsi_ma20:,.0f}", delta_color="off")
        st.markdown(f"**狀態：{light_trend} ({score_trend}/2分)**")
        st.caption(f"50天線：{hsi_ma50:,.0f}｜{desc_trend}")
    with c2:
        st.subheader("2. 藍籌市寬 (企穩50天線)")
        st.metric("藍籌企穩比例", f"{breadth_pct:.1f}%", f"{above_50} / {valid_stocks} 隻", delta_color="off")
        st.markdown(f"**狀態：{light_breadth} ({score_breadth}/2分)**")
        st.caption(">50%安全｜30-50%警戒｜<30%危險")
    with c3:
        st.subheader("3. 機構派發日 (20日內)")
        st.metric("2800.HK 派發日數", f"{dist_days} 日", "跌>=0.6%且成交增加", delta_color="off")
        st.markdown(f"**狀態：{light_inst} ({score_inst}/2分)**")
        st.caption(desc_inst)

    st.divider()
    st.subheader("恒生指數走勢與 20 天、50 天線")
    st.line_chart(hsi[["Close", "MA20", "MA50"]].tail(250).rename(columns={"Close": "恒指收市", "MA20": "20天線", "MA50": "50天線"}))

# ================= 分頁 2：歷史回測 =================
with tab2:
    st.subheader("📊 港股紅綠燈 (加速改良版)：歷史回測")
    st.caption("回測邏輯：以每日收市後計算之 3 大指標總分（0-6分），模擬動態倉位（綠燈100%、黃燈50%、紅燈0%空倉）對比「長揸恒指」。")

    if st.button("🚀 開始執行歷史回測", type="primary"):
        with st.spinner("正在進行回測計算，約需 10 秒..."):
            bt = run_backtest_data()

        start_d = bt.index[0].strftime("%Y-%m-%d")
        end_d = bt.index[-1].strftime("%Y-%m-%d")
        st.success(f"回測完成！區間：{start_d} 至 {end_d}（共 {len(bt)} 個交易日）")

        st.markdown("#### 1. 各評分區間下，恒指之後的平均表現")
        summary_rows = []
        for zone in ["🟢 安全區 (5-6分)", "🟡 警戒區 (3-4分)", "🔴 危險區 (0-2分)"]:
            sub = bt[bt["Zone"] == zone]
            if len(sub) > 0:
                avg_20 = sub["Fwd_20d"].mean() * 100
                win_20 = (sub["Fwd_20d"] > 0).mean() * 100
                avg_60 = sub["Fwd_60d"].mean() * 100
                win_60 = (sub["Fwd_60d"] > 0).mean() * 100
                summary_rows.append({
                    "評分區間": zone,
                    "出現日數": f"{len(sub)} 日",
                    "後20日平均升跌": f"{avg_20:+.2f}%",
                    "後20日勝率": f"{win_20:.1f}%",
                    "後60日平均升跌": f"{avg_60:+.2f}%",
                    "後60日勝率": f"{win_60:.1f}%",
                })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        st.markdown("#### 2. 動態倉位策略 vs 長揸恒指（起點 = 100）")
        bh_ret = bt["恒指長揸 (Buy & Hold)"].iloc[-1] - 100
        st_ret = bt["紅綠燈動態倉位策略"].iloc[-1] - 100

        def max_dd(series):
            return ((series / series.cummax()) - 1).min() * 100

        bh_dd = max_dd(bt["恒指長揸 (Buy & Hold)"])
        st_dd = max_dd(bt["紅綠燈動態倉位策略"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("紅綠燈策略總回報", f"{st_ret:+.1f}%")
        m2.metric("長揸恒指總回報", f"{bh_ret:+.1f}%")
        m3.metric("紅綠燈最大回撤 (風險)", f"{st_dd:.1f}%")
        m4.metric("長揸恒指最大回撤", f"{bh_dd:.1f}%")

        st.line_chart(bt[["恒指長揸 (Buy & Hold)", "紅綠燈動態倉位策略"]])
