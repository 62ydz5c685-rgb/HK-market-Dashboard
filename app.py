import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

st.set_page_config(page_title="港股量化雙核系統 (O'Neil 大市時機版)", page_icon="📈", layout="wide")

# 82 隻千億市值核心港股
HK_STOCKS_DICT = {
    # 科網 / 平台經濟
    "0700.HK": "騰訊控股", "9988.HK": "阿里巴巴", "3690.HK": "美團", "9618.HK": "京東集團",
    "9999.HK": "網易", "1024.HK": "快手", "9888.HK": "百度集團", "9961.HK": "攜程集團",
    "1810.HK": "小米集團", "0992.HK": "聯想集團", "2015.HK": "理想汽車",
    # 銀行 / 金融 / 保險
    "0005.HK": "滙豐控股", "1299.HK": "友邦保險", "0388.HK": "香港交易所", "1398.HK": "工商銀行",
    "0939.HK": "建設銀行", "3988.HK": "中國銀行", "1288.HK": "農業銀行", "3968.HK": "招商銀行",
    "1658.HK": "郵儲銀行", "3328.HK": "交通銀行", "2388.HK": "中銀香港", "2318.HK": "中國平安",
    "2628.HK": "中國人壽", "2328.HK": "中國財險", "6030.HK": "中信証券",
    # 電訊 / 公用 / 能源
    "0941.HK": "中國移動", "0728.HK": "中國電信", "0762.HK": "中國聯通", "0883.HK": "中國海洋石油",
    "0857.HK": "中國石油股份", "0386.HK": "中國石油化工", "1088.HK": "中國神華", "0002.HK": "中電控股",
    "0003.HK": "中華煤氣", "1038.HK": "長江基建集團", "0836.HK": "華潤電力",
    # 資源 / 礦產 / 原材料
    "2899.HK": "紫金礦業", "3993.HK": "洛陽鉬業", "1378.HK": "中國宏橋", "0358.HK": "江西銅業",
    "1898.HK": "中煤能源",
    # 汽車製造 / 半導體 / 硬件
    "1211.HK": "比亞迪股份", "0175.HK": "吉利汽車", "0300.HK": "美的集團", "6690.HK": "海爾智家",
    "0981.HK": "中芯國際", "2382.HK": "舜宇光學科技", "0669.HK": "創科實業",
    # 醫藥生物
    "6160.HK": "百濟神州", "1801.HK": "信達生物", "2359.HK": "藥明康德", "2269.HK": "藥明生物",
    "1093.HK": "石藥集團", "1177.HK": "中國生物製藥",
    # 消費零售 / 搏彩娛樂
    "9633.HK": "農夫山泉", "2020.HK": "安踏體育", "9992.HK": "泡泡瑪特", "0291.HK": "華潤啤酒",
    "0288.HK": "萬洲國際", "2319.HK": "蒙牛乳業", "2313.HK": "申洲國際", "1928.HK": "金沙中國",
    "0027.HK": "銀河娛樂",
    # 地產物業 / 物流航運 / 綜合企業
    "0001.HK": "長和", "0016.HK": "新鴻基地產", "0012.HK": "恒基地產", "1109.HK": "華潤置地",
    "0688.HK": "中國海外發展", "1113.HK": "長實集團", "0267.HK": "中信股份", "0066.HK": "港鐵公司",
    "0823.HK": "領展房產基金", "1919.HK": "中遠海控", "0960.HK": "龍湖集團"
}

# 1. 盈富基金估值數據
@st.cache_data(ttl=1800)
def fetch_tracker_data():
    ticker = yf.Ticker("2800.HK")
    # 長歷史 (算 5 年 SD 用) + 最近 5 日 (避開長歷史伺服器緩存延遲)
    raw_hist = ticker.history(start="2005-01-01", auto_adjust=False)
    raw_recent = ticker.history(period="5d", auto_adjust=False)
    raw = pd.concat([raw_hist, raw_recent])
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    raw.index = pd.to_datetime(raw.index).tz_localize(None)

    # 關鍵修復：若 Yahoo 尚未計算最新一天的 Adj Close，自動以 Close 填補，絕不刪除最新日
    if "Adj Close" in raw.columns:
        raw["Adj Close"] = raw["Adj Close"].fillna(raw["Close"])
    else:
        raw["Adj Close"] = raw["Close"]

    raw = raw.dropna(subset=["Close"])

    divs = raw[raw["Dividends"] > 0]["Dividends"]
    ttm_div = divs.rolling(2).sum().reindex(raw.index).ffill()

    df = pd.DataFrame(index=raw.index)
    df["Close"] = raw["Close"]
    df["Adj_Close"] = raw["Adj Close"]
    df["TTM_Div"] = ttm_div
    df["Yield"] = (df["TTM_Div"] / df["Close"]) * 100
    df = df.dropna(subset=["Yield"]).copy()

    df["Mean_5Y"] = df["Yield"].rolling("1825D", min_periods=252).mean()
    df["Std_5Y"] = df["Yield"].rolling("1825D", min_periods=252).std()
    df["Z"] = (df["Yield"] - df["Mean_5Y"]) / df["Std_5Y"]
    return df.dropna(subset=["Z"]).copy(), divs


# 2. William O'Neil 大市時機 (FTD & Distribution Days) 計算
@st.cache_data(ttl=3600)
def fetch_oneil_market_data():
    start_str = (datetime.now() - timedelta(days=700)).strftime("%Y-%m-%d")
    hsi = yf.Ticker("^HSI").history(start=start_str, auto_adjust=True)
    hsi.index = pd.to_datetime(hsi.index).tz_localize(None)
    
    # 若指數本身成交量缺失，則備用 2800.HK 真實成交量作為大市活躍度指標
    t2800 = yf.Ticker("2800.HK").history(start=start_str, auto_adjust=True)
    t2800.index = pd.to_datetime(t2800.index).tz_localize(None)
    
    if hsi["Volume"].sum() == 0 or hsi["Volume"].isna().all():
        hsi["Volume"] = t2800["Volume"].reindex(hsi.index).ffill()

    hsi["Ret"] = hsi["Close"].pct_change()
    hsi["MA50"] = hsi["Close"].rolling(50).mean()
    hsi["MA200"] = hsi["Close"].rolling(200).mean()

    # 出貨日 (Distribution Day) 判斷：收市跌 >= 0.2% 且 成交量大於前一日
    is_down_day = hsi["Ret"] <= -0.002
    is_vol_up = hsi["Volume"] > hsi["Volume"].shift(1)
    hsi["Is_DD"] = is_down_day & is_vol_up

    # 檢視近 25 個交易日內的所有出貨日及其有效性
    window_days = 25
    recent_25 = hsi.iloc[-window_days:].copy()
    dd_records = []
    
    curr_close = hsi["Close"].iloc[-1]
    
    for dt, row in recent_25.iterrows():
        if row["Is_DD"]:
            dd_price = row["Close"]
            days_ago = len(hsi.loc[dt:]) - 1
            # 檢查之後的收市價有無高於該日 5%
            max_after = hsi.loc[dt:, "Close"].max()
            erased_by_gain = max_after >= dd_price * 1.05
            
            status = "🚨 有效（壓制大市）"
            if erased_by_gain:
                status = "✅ 已消化（指數升穿 >5%）"
                
            dd_records.append({
                "日期": dt.strftime("%Y-%m-%d"),
                "距今交易日": f"{days_ago} 日前",
                "當日收市點數": f"{dd_price:.0f}",
                "當日跌幅": f"{row['Ret']*100:.2f}%",
                "當日成交量爆發": f"比前日升 {((row['Volume']/hsi['Volume'].shift(1).loc[dt])-1)*100:+.1f}%",
                "當前狀態": status,
                "is_active": not erased_by_gain
            })

    active_dd_count = sum(1 for r in dd_records if r["is_active"])

    # 跟進日 (Follow-Through Day, FTD) 檢測
    # 邏輯：跌浪過後首個向上日為 Day 1，隨後 Day 4-15 內出現單日大升 >= 1.2% 且 放量
    ftd_records = []
    for i in range(50, len(hsi)):
        # 尋找前期跌勢 (低於 20 日低位)
        sub = hsi.iloc[i-30:i+1]
        c_today = sub["Close"].iloc[-1]
        c_prev = sub["Close"].iloc[-2]
        vol_today = sub["Volume"].iloc[-1]
        vol_prev = sub["Volume"].iloc[-2]
        ret = (c_today / c_prev) - 1

        if ret >= 0.012 and vol_today > vol_prev:
            # 檢查過去 4-15 日內是否曾創波段低位並嘗試反彈
            past_lows = sub["Low"].iloc[-15:-3]
            if len(past_lows) > 0 and past_lows.min() < sub["Close"].iloc[-15]:
                dt_str = sub.index[-1].strftime("%Y-%m-%d")
                ftd_records.append({
                    "跟進日日期": dt_str,
                    "當日點數": f"{c_today:.0f}",
                    "當日升幅": f"+{ret*100:.2f}%",
                    "當日成交量增長": f"+{((vol_today/vol_prev)-1)*100:.1f}%",
                    "目前距今": f"{len(hsi)-1-i} 個交易日"
                })

    return hsi, dd_records, active_dd_count, ftd_records

# 3. 千億 VCP 雷達數據下載
@st.cache_data(ttl=3600)
def fetch_vcp_data():
    start_str = (datetime.now() - timedelta(days=500)).strftime("%Y-%m-%d")
    hsi = yf.Ticker("^HSI").history(start=start_str, auto_adjust=True)
    hsi.index = pd.to_datetime(hsi.index).tz_localize(None)
    hsi["MA50"] = hsi["Close"].rolling(50).mean()

    tickers = list(HK_STOCKS_DICT.keys())
    raw = yf.download(tickers, start=start_str, group_by="ticker", auto_adjust=True, progress=False, threads=True)

    breakout_list = []
    watchlist = []

    for tk, name in HK_STOCKS_DICT.items():
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if tk not in raw.columns.get_level_values(0):
                    continue
                d = raw[tk].dropna(subset=["Close", "Volume"]).copy()
            else:
                continue

            d.index = pd.to_datetime(d.index).tz_localize(None)
            if len(d) < 260:
                continue

            d["MA50"] = d["Close"].rolling(50).mean()
            d["MA150"] = d["Close"].rolling(150).mean()
            d["MA200"] = d["Close"].rolling(200).mean()
            d["High_252"] = d["High"].rolling(252).max()
            d["High_20_Prev"] = d["High"].rolling(20).max().shift(1)
            d["Vol_MA50"] = d["Volume"].rolling(50).mean()

            d["Range_10"] = (d["High"].rolling(10).max() - d["Low"].rolling(10).min()) / d["Close"]
            d["Range_50"] = (d["High"].rolling(50).max() - d["Low"].rolling(50).min()) / d["Close"]

            latest = d.iloc[-1]
            c = latest["Close"]
            ma50 = latest["MA50"]
            ma150 = latest["MA150"]
            ma200 = latest["MA200"]
            ma200_20d_ago = d["MA200"].iloc[-21] if len(d) >= 221 else ma200
            high_252 = latest["High_252"]
            high_20_prev = latest["High_20_Prev"]
            vol = latest["Volume"]
            vol_ma50 = latest["Vol_MA50"]
            range_10 = latest["Range_10"]
            range_50 = latest["Range_50"]
            rs_60 = (c / d["Close"].iloc[-61] - 1) * 100 if len(d) >= 61 else 0.0

            is_stage2 = (
                (c > ma50) and (ma50 > ma150) and (ma150 > ma200) and
                (ma200 > ma200_20d_ago) and (c >= high_252 * 0.75)
            )
            is_tight = (range_10 < 0.15) and (range_10 < range_50 * 0.70)
            is_breakout = (c > high_20_prev) and (vol > vol_ma50 * 1.3)

            vol_ratio = (vol / vol_ma50) if vol_ma50 > 0 else 0.0
            stop_loss = c * 0.93

            if is_stage2 and is_tight:
                if is_breakout:
                    breakout_list.append({
                        "代號": tk,
                        "股票名稱": name,
                        "最新收市價": f"${c:.2f}",
                        "建議 -7% 止蝕價": f"${stop_loss:.2f}",
                        "50MA 離場防線": f"${ma50:.2f}",
                        "突破幅度": f"+{((c - high_20_prev) / high_20_prev) * 100:.1f}%",
                        "成交量爆發": f"{vol_ratio:.1f}x 均量",
                        "60日相對強度 (RS)": f"{rs_60:+.1f}%",
                        "rs_val": rs_60
                    })
                elif c <= high_20_prev:
                    dist_to_breakout = ((high_20_prev - c) / c) * 100
                    watchlist.append({
                        "代號": tk,
                        "股票名稱": name,
                        "最新現價": f"${c:.2f}",
                        "突破目標價 (前高)": f"${high_20_prev:.2f}",
                        "距離突破仲差": f"{dist_to_breakout:.1f}%",
                        "近10日波幅 (收縮度)": f"{range_10 * 100:.1f}%",
                        "50MA 基準": f"${ma50:.2f}",
                        "今日量比": f"{vol_ratio:.2f}x 均量",
                        "60日相對強度 (RS)": f"{rs_60:+.1f}%",
                        "dist_val": dist_to_breakout
                    })
        except Exception:
            continue

    breakout_list.sort(key=lambda x: x["rs_val"], reverse=True)
    watchlist.sort(key=lambda x: x["dist_val"])
    return hsi, breakout_list, watchlist

# 標題與更新掣
col_title, col_btn = st.columns([4, 1])
with col_title:
    st.title("📈 港股雙核量化系統")
    hk_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    st.caption(f"香港時間更新：{hk_time}｜核心底倉 (2800.HK 息率 SD) ＋ 大市環境 (O'Neil FTD/派發日) ＋ 進攻衛星 (千億 VCP)")
with col_btn:
    st.write("")
    if st.button("🔄 立即更新數據", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# 4 個獨立主分頁
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 2800.HK 今日估值與階梯",
    "📊 2800.HK 零上帝視角回測",
    "🚦 O'Neil 大市時機 (FTD / 出貨日)",
    "🚀 千億港股 VCP 突破雷達"
])

# ================= 頁面 1：2800.HK 今日估值與階梯 =================
with tab1:
    with st.spinner("正在載入盈富基金歷史數據..."):
        df_2800, divs = fetch_tracker_data()

    latest = df_2800.iloc[-1]
    latest_date = df_2800.index[-1].strftime("%Y-%m-%d")
    latest_price = latest["Close"]
    latest_div = latest["TTM_Div"]
    latest_yield = latest["Yield"]
    mean_5y = latest["Mean_5Y"]
    std_5y = latest["Std_5Y"]
    latest_z = latest["Z"]

    def get_tier_info(z):
        if z >= 2.0:
            return "100% 滿倉 (5/5 倉)", "🟢 極度便宜 (+2SD 以上)", "success", "息率高於近5年均值逾兩個標準差，安全邊際極強，建議滿倉。"
        elif z >= 1.0:
            return "80% 倉位 (4/5 倉)", "🟢 偏便宜 (+1SD 至 +2SD)", "success", "息率具明顯吸引力，屬於價值低吸區，維持重倉收息兼等反彈。"
        elif z >= 0.0:
            return "60% 倉位 (3/5 倉)", "🟡 略便宜 (均值至 +1SD)", "info", "估值略低於近5年平均，適合維持中高倉位穩健持貨。"
        elif z >= -1.0:
            return "40% 倉位 (2/5 倉)", "🟡 略貴 (-1SD 至均值)", "warning", "估值略高於近5年平均，建議偏防守配置，保留一半以上現金。"
        elif z >= -2.0:
            return "20% 倉位 (1/5 倉)", "🔴 偏貴 (-2SD 至 -1SD)", "error", "息率跌至近5年偏低水平，市場情緒偏熱，建議分批獲利減至輕倉。"
        else:
            return "0% 清倉 (0/5 倉)", "🔴 極度昂貴 (-2SD 以下)", "error", "息率處於近5年極低泡沫區，全面獲利離場，現金為王。"

    pos_text, zone_label, alert_type, advice = get_tier_info(latest_z)
    banner_msg = f"### 今日建議倉位：{pos_text} — {zone_label}\n{advice}（最後交易日：{latest_date}）"
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
    c2.metric("最新股息率", f"{latest_yield:.2f}%", f"近5年均值: {mean_5y:.2f}%", delta_color="off")
    c3.metric("標準差位置 (Z-Score)", f"{latest_z:+.2f} SD", "正數=偏平｜負數=偏貴", delta_color="off")
    c4.metric("過去12個月總派息", f"${latest_div:.2f}", "最近2次派息合計", delta_color="off")

    st.divider()

    st.subheader("📋 6 級息率階梯與 2800.HK 參考價（滾動 5 年均值）")
    y_p2 = mean_5y + 2 * std_5y
    y_p1 = mean_5y + std_5y
    y_m1 = mean_5y - std_5y
    y_m2 = mean_5y - 2 * std_5y

    ladder_df = pd.DataFrame([
        {"估值區間": "🟢 極度便宜 (> +2 SD)", "建議倉位": "100% (5/5 倉)", "觸發息率": f"≥ {y_p2:.2f}%", "參考價": f"${latest_div/(y_p2/100):.2f} 或以下", "目前位置": "👈 今日位置" if latest_z >= 2.0 else ""},
        {"估值區間": "🟢 偏便宜 (+1 至 +2 SD)", "建議倉位": "80% (4/5 倉)", "觸發息率": f"{y_p1:.2f}% – {y_p2:.2f}%", "參考價": f"${latest_div/(y_p2/100):.2f} – ${latest_div/(y_p1/100):.2f}", "目前位置": "👈 今日位置" if 1.0 <= latest_z < 2.0 else ""},
        {"估值區間": "🟡 略便宜 (Mean 至 +1 SD)", "建議倉位": "60% (3/5 倉)", "觸發息率": f"{mean_5y:.2f}% – {y_p1:.2f}%", "參考價": f"${latest_div/(y_p1/100):.2f} – ${latest_div/(mean_5y/100):.2f}", "目前位置": "👈 今日位置" if 0.0 <= latest_z < 1.0 else ""},
        {"估值區間": "🟡 略貴 (-1 SD 至 Mean)", "建議倉位": "40% (2/5 倉)", "觸發息率": f"{y_m1:.2f}% – {mean_5y:.2f}%", "參考價": f"${latest_div/(mean_5y/100):.2f} – ${latest_div/(y_m1/100):.2f}", "目前位置": "👈 今日位置" if -1.0 <= latest_z < 0.0 else ""},
        {"估值區間": "🔴 偏貴 (-2 至 -1 SD)", "建議倉位": "20% (1/5 倉)", "觸發息率": f"{y_m2:.2f}% – {y_m1:.2f}%", "參考價": f"${latest_div/(y_m1/100):.2f} – ${latest_div/(y_m2/100):.2f}", "目前位置": "👈 今日位置" if -2.0 <= latest_z < -1.0 else ""},
        {"估值區間": "🔴 極度昂貴 (< -2 SD)", "建議倉位": "0% (0/5 倉)", "觸發息率": f"< {y_m2:.2f}%", "參考價": f"${latest_div/(y_m2/100):.2f} 或以上", "目前位置": "👈 今日位置" if latest_z < -2.0 else ""},
    ])
    st.dataframe(ladder_df, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("📈 盈富基金股息率與「動態 5 年 SD 通道」走勢 (%)")
    chart_df = pd.DataFrame(index=df_2800.index)
    chart_df["2800.HK 股息率 (%)"] = df_2800["Yield"]
    chart_df["+2SD (滿倉線)"] = df_2800["Mean_5Y"] + 2 * df_2800["Std_5Y"]
    chart_df["+1SD (80%線)"] = df_2800["Mean_5Y"] + df_2800["Std_5Y"]
    chart_df["滾動5年均值"] = df_2800["Mean_5Y"]
    chart_df["-1SD (20%線)"] = df_2800["Mean_5Y"] - df_2800["Std_5Y"]
    chart_df["-2SD (清倉線)"] = df_2800["Mean_5Y"] - 2 * df_2800["Std_5Y"]
    st.line_chart(chart_df)

# ================= 頁面 2：2800.HK 零上帝視角回測 =================
with tab2:
    st.subheader("📊 滾動 5 年息率 SD 策略：零上帝視角回測（含股息再投資）")
    st.caption("每日只使用「截至當日為止已發生嘅過去 5 年數據」計算標準差與倉位，完全排除未來函數。")

    opt1, opt2 = st.columns(2)
    with opt1:
        rebal_freq = st.radio(
            "調倉頻率設定",
            ["每月月底調倉一次（推薦：大幅慳手續費）", "每日收市即時調倉"],
            horizontal=False
        )
    with opt2:
        cash_rate = st.slider(
            "空倉現金年息假設（例如擺高息活期／定期／貨幣基金 %）",
            min_value=0.0, max_value=5.0, value=2.5, step=0.5
        )

    bt = df_2800.copy()
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

# ================= 頁面 3：William O'Neil 大市時機 (FTD / 出貨日) =================
with tab3:
    st.subheader("🚦 William O'Neil 大市狀態監察器（CAN SLIM 核心時機）")
    st.caption("透過恒生指數（^HSI）過去 25 個交易日之機構出貨日（Distribution Days）總數及跟進日（Follow-Through Day, FTD）判斷市場頂底。")

    with st.spinner("正在分析恒指機構出貨日與 FTD 訊號..."):
        hsi_on, dd_list, active_dd, ftd_list = fetch_oneil_market_data()

    latest_h = hsi_on.iloc[-1]
    h_close = latest_h["Close"]
    h_ma50 = latest_h["MA50"]
    h_ma200 = latest_h["MA200"]

    # 決定 O'Neil 大市燈號
    if active_dd <= 2:
        mkt_status = "🟢 確認升勢 (Confirmed Uptrend)"
        mkt_desc = "過去 25 日只有 0–2 個有效出貨日，市場由買盤主導，機構並未大舉派貨。適合積極參與 VCP 個股突破。"
        banner_fn = st.success
    elif 3 <= active_dd <= 4:
        mkt_status = "🟡 升勢受壓 (Uptrend Under Pressure)"
        mkt_desc = f"過去 25 日錄得 {active_dd} 個有效出貨日，機構資金開始減磅。建議停止激進追高，將獲利持倉止蝕位收緊至 50MA 保本。"
        banner_fn = st.warning
    else:
        mkt_status = "🔴 市場修正 / 跌勢 (Market in Correction)"
        mkt_desc = f"過去 25 日錄得高達 {active_dd} 個有效出貨日，大戶全面派發。高機率出現持續回調，嚴格禁止追買任何突破，宜保留大量現金。"
        banner_fn = st.error

    banner_fn(f"### 當前大市時機狀態：{mkt_status}\n{mkt_desc}")

    o1, o2, o3, o4 = st.columns(4)
    o1.metric("恒指最新收市", f"{h_close:.0f} 點")
    o2.metric("近25日有效出貨日 (DD)", f"{active_dd} 個", "≤2 健康 ｜ 3-4 受壓 ｜ ≥5 危險", delta_color="off")
    o3.metric("50天均線 (中期趨勢)", f"{h_ma50:.0f} 點", f"{((h_close/h_ma50)-1)*100:+.1f}%", delta_color="off")
    o4.metric("200天均線 (牛熊分界)", f"{h_ma200:.0f} 點", f"{((h_close/h_ma200)-1)*100:+.1f}%", delta_color="off")

    st.divider()

    col_dd, col_ftd = st.columns([3, 2])

    with col_dd:
        st.subheader("📦 過去 25 個交易日出貨日（Distribution Days）紀錄")
        st.caption("定義：單日跌幅 ≥ 0.2% 且成交量大於前一日。若指數隨後反彈超越該日收市價 5%，則自動標記為「已消化」。")
        if dd_list:
            df_dd = pd.DataFrame(dd_list).drop(columns=["is_active"])
            st.dataframe(df_dd, use_container_width=True, hide_index=True)
        else:
            st.info("過去 25 個交易日沒有出現任何符合條件之出貨日，大市處於極健康狀態！")

    with col_ftd:
        st.subheader("🎯 最近跟進日（Follow-Through Day）歷史")
        st.caption("定義：跌浪後反彈第 4–15 日內，指數單日放量急升 ≥ 1.2%，標誌熊市見底轉強。")
        if ftd_list:
            df_ftd = pd.DataFrame(ftd_list[-6:]).iloc[::-1]  # 取最近 6 次
            st.dataframe(df_ftd, use_container_width=True, hide_index=True)
        else:
            st.info("近期暫無新 FTD 訊號。")

    with st.expander("📖 William O'Neil 大市時機法則（實戰操作備忘）"):
        st.markdown("""
        * **出貨日累積原理**：大市往往喺頂部依然一片欣欣向榮，但出貨日會提早反映機構暗中派貨。當出貨日喺 4–5 週內累積到 **5 個**，大盤見頂機率超過 80%。
        * **出貨日點樣消除？**：
          1. 滿 25 個交易日自然過期。
          2. 指數自出貨日收市價起向上升幅達 **5%**，該出貨日即時作廢（代表買盤徹底擊潰賣盤）。
        * **跟進日（FTD）操作**：大跌市千祈唔好左側亂估底；耐心等待出現放量急升的 FTD，翌日方可作為「第一注試水溫」開倉信號。
        """)

# ================= 頁面 4：千億港股 VCP 突破雷達 =================
with tab4:
    with st.spinner("正在掃描 82 隻千億市值港股並運算 VCP 形態（約需 10–15 秒）..."):
        hsi, breakouts, watch_list = fetch_vcp_data()

    latest_hsi = hsi.iloc[-1]
    hsi_close = latest_hsi["Close"]
    hsi_ma50 = latest_hsi["MA50"]
    hsi_diff_pct = ((hsi_close - hsi_ma50) / hsi_ma50) * 100
    mkt_is_green = hsi_close > hsi_ma50

    st.subheader("🚦 恒生指數 50MA 大市濾網狀態")
    if mkt_is_green:
        st.success(
            f"### 🟢 大市綠燈（環境良好，允許追突破開新倉）\n"
            f"恒指最新收市：**{hsi_close:.0f} 點** ｜ 50天線：**{hsi_ma50:.0f} 點**（企穩均線上方 **+{hsi_diff_pct:.2f}%**）"
        )
    else:
        st.error(
            f"### 🔴 大市紅燈（大市走弱，嚴格禁止追突破開新倉）\n"
            f"恒指最新收市：**{hsi_close:.0f} 點** ｜ 50天線：**{hsi_ma50:.0f} 點**（跌穿均線 **{hsi_diff_pct:.2f}%**）。已持倉股票請嚴格跟足 50MA 或 -7% 防線執行離場。"
        )

    st.divider()

    # 1. 今日放量突破
    st.subheader(f"🎯 今日放量突破股票（共 {len(breakouts)} 隻）")
    st.caption("符合條件：Minervini 第二階段趨勢 ＋ VCP 波幅收縮完成 ＋ 今日收市價突破過去 20 日高位 ＋ 成交量 > 1.3 倍 50 日均量。")

    if breakouts:
        b_df = pd.DataFrame(breakouts).drop(columns=["rs_val"])
        st.dataframe(b_df, use_container_width=True, hide_index=True)
    else:
        st.info("今日暫無千億市值股票觸發「放量突破」買入訊號。")

    st.divider()

    # 2. 形態收縮就緒（觀察名單）
    st.subheader(f"👀 形態收縮就緒・即將突破觀察名單（共 {len(watch_list)} 隻）")
    st.caption("符合條件：第二階段趨勢向上 ＋ 近 10 日波幅收縮至 15% 內且乾量洗盤完成，但尚未突破前高。按「距離突破差幅」由近至遠排序。")

    if watch_list:
        w_df = pd.DataFrame(watch_list).drop(columns=["dist_val"])
        st.dataframe(w_df, use_container_width=True, hide_index=True)
    else:
        st.info("今日暫無千億股票處於「VCP 收縮就緒」狀態。")

    with st.expander("📖 實戰操作與風控紀律說明（50MA 止賺 ＋ -7% 硬止蝕）"):
        st.markdown("""
        * **1. 大市綠燈才開倉**：只有當上方恒指呈現 **🟢 綠燈**（或 O'Neil 分頁處於確認升勢）時，先准按「今日放量突破名單」買入，最多同時持有 5 隻（每隻注碼 20%）。
        * **2. 觀察名單點樣用？**：睇實「距離突破仲差」低於 3% 嗰幾隻。翌日盤中如果見股價衝破「突破目標價」，且下午成交量明顯放大，尾市（15:55）即可準備入場。
        * **3. 買入即掛 `-7%` 硬止蝕**：按表格提供的「建議 -7% 止蝕價」喺券商掛條件單，盤中掂到無條件即走，保命第一。
        * **4. 50MA 移動止賺離場**：每日下午 4:00 收市檢視，若**收市價正式跌穿 50MA**，翌日開市離場鎖定利潤；未跌穿則安心持貨食盡大浪。
        """)
