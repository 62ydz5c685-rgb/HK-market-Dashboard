import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# 網頁基本設定
st.set_page_config(page_title="港股大市紅綠燈", page_icon="🚦", layout="wide")

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
    hyg = yf.Ticker("HYG").history(period="1y")
    ief = yf.Ticker("IEF").history(period="1y")
    breadth_data = yf.download(
        HSI_STOCKS, period="6mo", group_by="ticker", progress=False, threads=True
    )
    return hsi, tracker, hyg, ief, breadth_data

@st.cache_data(ttl=86400)
def run_backtest_data():
    # 下載 4 年數據（首 1 年用來預熱 200 天線，實際回測約 3 年）
    hsi = yf.Ticker("^HSI").history(period="4y")
    tracker = yf.Ticker("2800.HK").history(period="4y")
    hyg = yf.Ticker("HYG").history(period="4y")
    ief = yf.Ticker("IEF").history(period="4y")
    stocks_close = yf.download(HSI_STOCKS, period="4y", progress=False, threads=True)["Close"]

    # 統一移除時區以便對齊日期
    for d in [hsi, tracker, hyg, ief, stocks_close]:
        d.index = pd.to_datetime(d.index).tz_localize(None)

    df = pd.DataFrame(index=hsi.index)
    df["HSI"] = hsi["Close"]
    df["MA50"] = df["HSI"].rolling(50).mean()
    df["MA200"] = df["HSI"].rolling(200).mean()

    # 1. 趨勢評分
    df["S_Trend"] = np.where(
        (df["HSI"] > df["MA50"]) & (df["MA50"] > df["MA200"]), 2,
        np.where((df["HSI"] < df["MA200"]) & (df["MA50"] < df["MA200"]), 0, 1)
    )

    # 2. 市寬評分
    stocks_aligned = stocks_close.reindex(df.index).ffill()
    stocks_ma50 = stocks_aligned.rolling(50).mean()
    valid_cnt = stocks_ma50.notna().sum(axis=1)
    above_cnt = (stocks_aligned > stocks_ma50).sum(axis=1)
    df["Breadth"] = (above_cnt / valid_cnt * 100).fillna(0)
    df["S_Breadth"] = np.where(df["Breadth"] > 50, 2, np.where(df["Breadth"] >= 30, 1, 0))

    # 3. 債市避險評分 (HYG / IEF)
    ratio = (hyg["Close"] / ief["Close"]).dropna()
    bond_df = pd.DataFrame({"Ratio": ratio})
    bond_df["MA20"] = bond_df["Ratio"].rolling(20).mean()
    bond_df["MA50"] = bond_df["Ratio"].rolling(50).mean()
    bond_aligned = bond_df.reindex(df.index).ffill()
    df["S_Bond"] = np.where(
        (bond_aligned["Ratio"] > bond_aligned["MA20"]) & (bond_aligned["Ratio"] > bond_aligned["MA50"]), 2,
        np.where((bond_aligned["Ratio"] < bond_aligned["MA20"]) & (bond_aligned["Ratio"] < bond_aligned["MA50"]), 0, 1)
    )

    # 4. 機構動向評分 (2800.HK 25日派發日)
    tr = tracker.reindex(df.index).ffill()
    dist_day = (tr["Close"].pct_change() <= -0.002) & (tr["Volume"] > tr["Volume"].shift(1))
    df["Dist_Days"] = dist_day.rolling(25).sum()
    df["S_Inst"] = np.where(df["Dist_Days"] <= 2, 2, np.where(df["Dist_Days"] <= 4, 1, 0))

    # 總分與分區
    df["Score"] = df["S_Trend"] + df["S_Breadth"] + df["S_Bond"] + df["S_Inst"]
    df = df.dropna(subset=["MA200"]).copy()

    df["Zone"] = np.where(
        df["Score"] >= 6, "🟢 安全區 (6-8分)",
        np.where(df["Score"] >= 3, "🟡 警戒區 (3-5分)", "🔴 危險區 (0-2分)")
    )

    # 計算未來 20 日及 60 日回報
    df["Ret_1d"] = df["HSI"].pct_change().fillna(0)
    df["Fwd_20d"] = df["HSI"].shift(-20) / df["HSI"] - 1
    df["Fwd_60d"] = df["HSI"].shift(-60) / df["HSI"] - 1

    # 模擬策略：用「前一日收市評分」決定「今日倉位」（避免未來函數）
    pos = np.where(df["Score"] >= 6, 1.0, np.where(df["Score"] >= 3, 0.5, 0.0))
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

# 標題與手動更新掣
col_title, col_btn = st.columns([4, 1])
with col_title:
    st.title("🚦 港股大市紅綠燈")
    hk_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    st.caption(f"香港時間更新：{hk_time}（建議每日 16:30 收市後檢視）")
with col_btn:
    st.write("")
    if st.button("🔄 立即更新數據", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# 建立兩個分頁
tab1, tab2 = st.tabs(["🚦 今日大市體檢", "📊 3年歷史回測 (Backtest)"])

# ================= 分頁 1：今日大市體檢 =================
with tab1:
    with st.spinner("正在從 Yahoo Finance 下載最新收市數據，請稍候約 10 秒..."):
        hsi, tracker, hyg, ief, breadth_data = fetch_market_data()

    hsi["MA50"] = hsi["Close"].rolling(50).mean()
    hsi["MA200"] = hsi["Close"].rolling(200).mean()
    hsi_close, hsi_ma50, hsi_ma200 = hsi["Close"].iloc[-1], hsi["MA50"].iloc[-1], hsi["MA200"].iloc[-1]
    hsi_date = hsi.index[-1].strftime("%Y-%m-%d")

    if hsi_close > hsi_ma50 and hsi_ma50 > hsi_ma200:
        score_trend, light_trend, desc_trend = 2, "🟢 安全", "收市高於50天線，且50天線高於200天線"
    elif hsi_close < hsi_ma200 and hsi_ma50 < hsi_ma200:
        score_trend, light_trend, desc_trend = 0, "🔴 危險", "收市跌穿200天線，且50天線低於200天線"
    else:
        score_trend, light_trend, desc_trend = 1, "🟡 警戒", "處於50天與200天線之間，趨勢未明"

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

    bond_df = pd.concat([hyg["Close"], ief["Close"]], axis=1, keys=["HYG", "IEF"]).dropna()
    bond_df["Ratio"] = bond_df["HYG"] / bond_df["IEF"]
    bond_df["MA20"] = bond_df["Ratio"].rolling(20).mean()
    bond_df["MA50"] = bond_df["Ratio"].rolling(50).mean()
    ratio_now, ratio_ma20, ratio_ma50 = bond_df["Ratio"].iloc[-1], bond_df["MA20"].iloc[-1], bond_df["MA50"].iloc[-1]

    if ratio_now > ratio_ma20 and ratio_now > ratio_ma50:
        score_bond, light_bond, desc_bond = 2, "🟢 安全", "比率高於20天及50天線，資金願意承險"
    elif ratio_now < ratio_ma20 and ratio_now < ratio_ma50:
        score_bond, light_bond, desc_bond = 0, "🔴 危險", "比率跌穿20天及50天線，資金轉入國債避險"
    else:
        score_bond, light_bond, desc_bond = 1, "🟡 警戒", "比率處於20天與50天線之間，資金現分歧"

    tracker["Pct_Change"] = tracker["Close"].pct_change()
    tracker["Vol_Up"] = tracker["Volume"] > tracker["Volume"].shift(1)
    tracker["Dist_Day"] = (tracker["Pct_Change"] <= -0.002) & tracker["Vol_Up"]
    dist_days = int(tracker.tail(25)["Dist_Day"].sum())

    if dist_days <= 2:
        score_inst, light_inst, desc_inst = 2, "🟢 安全", "大戶未見明顯出貨跡象"
    elif dist_days <= 4:
        score_inst, light_inst, desc_inst = 1, "🟡 警戒", "開始出現大戶逢高派貨"
    else:
        score_inst, light_inst, desc_inst = 0, "🔴 危險", "機構密集撤退，見頂風險高"

    rally_day, rally_low, ftd_info = detect_ftd(tracker)
    total_score = score_trend + score_breadth + score_bond + score_inst

    if total_score >= 6:
        st.success(f"### 總評分：{total_score} / 8 分 — 🟢 安全持貨區\n大市趨勢、市寬與環球資金配合，可安心持貨並按正常策略操作。（恒指最後交易日：{hsi_date}）")
    elif total_score >= 3:
        st.warning(f"### 總評分：{total_score} / 8 分 — 🟡 提高警覺區\n市場出現轉弱或分歧訊號。建議停止開新倉、收緊止蝕位，並考慮適度減磅。（恒指最後交易日：{hsi_date}）")
    else:
        st.error(f"### 總評分：{total_score} / 8 分 — 🔴 高度危險區\n系統性風險偏高，建議以現金為主，嚴控倉位，避免隨便撈底。（恒指最後交易日：{hsi_date}）")

    if ftd_info:
        st.success(f"🚀 **O'Neil 跟進日 (FTD) 已確認！**｜盈富基金 (2800.HK) 於 **{ftd_info['date']}**（反彈第 {ftd_info['day']} 日）放量上升 **+{ftd_info['gain']:.2f}%**，且至今未跌穿反彈防守低位 **{rally_low:.2f}**，可留意轉勢機會。")
    elif rally_day >= 4:
        st.warning(f"⏳ **O'Neil 跟進日監察中（嘗試反彈第 {rally_day} 日）**｜2800.HK 暫守住反彈低位 **{rally_low:.2f}**，正等待單日升 ≥1.25% 且成交大於前一日的確認訊號。")
    elif rally_day >= 1:
        st.info(f"👀 **O'Neil 嘗試反彈初期（第 {rally_day} 日）**｜2800.HK 剛於低位 **{rally_low:.2f}** 出現止跌跡象，需待第 4 日或之後才可確認跟進日。")
    else:
        st.info(f"⚪ **O'Neil 跟進日狀態：未見反彈訊號**｜2800.HK 剛創近期低位（{rally_low:.2f}）或尚未出現止跌日，建議耐心觀望。")

    if breadth_pct < 15:
        st.info("🟣 **極度超賣提示**：目前市寬低於 15%，代表跌勢已相當極端，短期容易出現技術反彈，切忌在此位置恐慌斬倉。")

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.subheader("1. 大市趨勢")
        st.metric("恒生指數 (^HSI)", f"{hsi_close:,.0f}", f"50天線: {hsi_ma50:,.0f}", delta_color="off")
        st.markdown(f"**狀態：{light_trend} ({score_trend}/2分)**")
        st.caption(f"200天線：{hsi_ma200:,.0f}｜{desc_trend}")
    with c2:
        st.subheader("2. 港股市寬")
        st.metric("企穩 50 天線藍籌比例", f"{breadth_pct:.1f}%", f"{above_50} / {valid_stocks} 隻", delta_color="off")
        st.markdown(f"**狀態：{light_breadth} ({score_breadth}/2分)**")
        st.caption(">50%安全｜30-50%警戒｜<30%危險")
    with c3:
        st.subheader("3. 債市避險度")
        st.metric("HYG ÷ IEF 比率", f"{ratio_now:.4f}", f"50天線: {ratio_ma50:.4f}", delta_color="off")
        st.markdown(f"**狀態：{light_bond} ({score_bond}/2分)**")
        st.caption(desc_bond)
    with c4:
        st.subheader("4. 機構動向")
        st.metric("2800.HK 派發日 (25日內)", f"{dist_days} 日", "跌>0.2%且成交增加", delta_color="off")
        st.markdown(f"**狀態：{light_inst} ({score_inst}/2分)**")
        st.caption(desc_inst)

    st.divider()
    chart1, chart2 = st.columns(2)
    with chart1:
        st.subheader("恒生指數近 1 年走勢與移動平均線")
        st.line_chart(hsi[["Close", "MA50", "MA200"]].tail(250).rename(columns={"Close": "恒指收市", "MA50": "50天線", "MA200": "200天線"}))
    with chart2:
        st.subheader("債市避險比率 (HYG / IEF) 近 1 年走勢")
        st.line_chart(bond_df[["Ratio", "MA20", "MA50"]].tail(250).rename(columns={"Ratio": "HYG/IEF 比率", "MA20": "20天線", "MA50": "50天線"}))

# ================= 分頁 2：3年歷史回測 =================
with tab2:
    st.subheader("📊 港股紅綠燈：過去 3 年歷史回測")
    st.caption("測試邏輯：以每日收市後計出之總分，觀察之後 20 日（約1個月）及 60 日（約3個月）恒指表現；並模擬「綠燈100%倉、黃燈50%倉、紅燈0%空倉」與「長揸恒指」之對比。")

    if st.button("🚀 開始執行 3 年歷史回測", type="primary"):
        with st.spinner("正在下載過去 4 年數據並計算每日歷史評分，約需 15–20 秒..."):
            bt = run_backtest_data()

        start_d = bt.index[0].strftime("%Y-%m-%d")
        end_d = bt.index[-1].strftime("%Y-%m-%d")
        st.success(f"回測完成！數據區間：{start_d} 至 {end_d}（共 {len(bt)} 個交易日）")

        # 1. 各評分區間表現表
        st.markdown("#### 1. 不同評分區間下，恒指之後嘅表現")
        summary_rows = []
        for zone in ["🟢 安全區 (6-8分)", "🟡 警戒區 (3-5分)", "🔴 危險區 (0-2分)"]:
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

        # 2. 策略淨值曲線與最大回撤
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
        st.caption("註：回測未計及交易手續費及股息；藍籌市寬以現時成份股回溯計算，僅供策略驗證參考。")
