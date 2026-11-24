import streamlit as st
import yfinance as yf
import pandas as pd
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
    # 1. 恒生指數 (^HSI)
    hsi = yf.Ticker("^HSI").history(period="2y")
    
    # 2. 盈富基金 (2800.HK)
    tracker = yf.Ticker("2800.HK").history(period="6mo")
    
    # 3. 美債避險比率 (HYG / IEF)
    hyg = yf.Ticker("HYG").history(period="1y")
    ief = yf.Ticker("IEF").history(period="1y")
    
    # 4. 藍籌股市寬數據
    breadth_data = yf.download(
        HSI_STOCKS, period="6mo", group_by="ticker", progress=False, threads=True
    )
    
    return hsi, tracker, hyg, ief, breadth_data

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

with st.spinner("正在從 Yahoo Finance 下載最新收市數據，請稍候約 10 秒..."):
    hsi, tracker, hyg, ief, breadth_data = fetch_market_data()

# --- 計算指標 1：大市趨勢 (^HSI) ---
hsi["MA50"] = hsi["Close"].rolling(50).mean()
hsi["MA200"] = hsi["Close"].rolling(200).mean()
hsi_close = hsi["Close"].iloc[-1]
hsi_ma50 = hsi["MA50"].iloc[-1]
hsi_ma200 = hsi["MA200"].iloc[-1]
hsi_date = hsi.index[-1].strftime("%Y-%m-%d")

if hsi_close > hsi_ma50 and hsi_ma50 > hsi_ma200:
    score_trend, light_trend, desc_trend = 2, "🟢 安全", "收市高於50天線，且50天線高於200天線"
elif hsi_close < hsi_ma200 and hsi_ma50 < hsi_ma200:
    score_trend, light_trend, desc_trend = 0, "🔴 危險", "收市跌穿200天線，且50天線低於200天線"
else:
    score_trend, light_trend, desc_trend = 1, "🟡 警戒", "處於50天與200天線之間，趨勢未明"

# --- 計算指標 2：港股市寬 (企穩50天線比例) ---
above_50 = 0
valid_stocks = 0
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

# --- 計算指標 3：債市避險度 (HYG / IEF) ---
bond_df = pd.concat([hyg["Close"], ief["Close"]], axis=1, keys=["HYG", "IEF"]).dropna()
bond_df["Ratio"] = bond_df["HYG"] / bond_df["IEF"]
bond_df["MA20"] = bond_df["Ratio"].rolling(20).mean()
bond_df["MA50"] = bond_df["Ratio"].rolling(50).mean()

ratio_now = bond_df["Ratio"].iloc[-1]
ratio_ma20 = bond_df["MA20"].iloc[-1]
ratio_ma50 = bond_df["MA50"].iloc[-1]

if ratio_now > ratio_ma20 and ratio_now > ratio_ma50:
    score_bond, light_bond, desc_bond = 2, "🟢 安全", "比率高於20天及50天線，資金願意承險"
elif ratio_now < ratio_ma20 and ratio_now < ratio_ma50:
    score_bond, light_bond, desc_bond = 0, "🔴 危險", "比率跌穿20天及50天線，資金轉入國債避險"
else:
    score_bond, light_bond, desc_bond = 1, "🟡 警戒", "比率處於20天與50天線之間，資金現分歧"

# --- 計算指標 4：機構動向 (2800.HK 過去25日派發日) ---
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

# --- 總評分顯示 ---
total_score = score_trend + score_breadth + score_bond + score_inst

st.divider()
if total_score >= 6:
    st.success(f"### 總評分：{total_score} / 8 分 — 🟢 安全持貨區\n大市趨勢、市寬與環球資金配合，可安心持貨並按正常策略操作。（恒指最後交易日：{hsi_date}）")
elif total_score >= 3:
    st.warning(f"### 總評分：{total_score} / 8 分 — 🟡 提高警覺區\n市場出現轉弱或分歧訊號。建議停止開新倉、收緊止蝕位，並考慮適度減磅。（恒指最後交易日：{hsi_date}）")
else:
    st.error(f"### 總評分：{total_score} / 8 分 — 🔴 高度危險區\n系統性風險偏高，建議以現金為主，嚴控倉位，避免隨便撈底。（恒指最後交易日：{hsi_date}）")

if breadth_pct < 15:
    st.info("🟣 **極度超賣提示**：目前市寬低於 15%，代表跌勢已相當極端，短期容易出現技術反彈，切忌在此位置恐慌斬倉。")

st.divider()

# --- 四大指標詳細卡片 ---
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.subheader("1. 大市趨勢")
    st.metric(label="恒生指數 (^HSI)", value=f"{hsi_close:,.0f}", delta=f"50天線: {hsi_ma50:,.0f}", delta_color="off")
    st.markdown(f"**狀態：{light_trend} ({score_trend}/2分)**")
    st.caption(f"200天線：{hsi_ma200:,.0f}｜{desc_trend}")

with c2:
    st.subheader("2. 港股市寬")
    st.metric(label="企穩 50 天線藍籌比例", value=f"{breadth_pct:.1f}%", delta=f"{above_50} / {valid_stocks} 隻", delta_color="off")
    st.markdown(f"**狀態：{light_breadth} ({score_breadth}/2分)**")
    st.caption(">50%安全｜30-50%警戒｜<30%危險")

with c3:
    st.subheader("3. 債市避險度")
    st.metric(label="HYG ÷ IEF 比率", value=f"{ratio_now:.4f}", delta=f"50天線: {ratio_ma50:.4f}", delta_color="off")
    st.markdown(f"**狀態：{light_bond} ({score_bond}/2分)**")
    st.caption(desc_bond)

with c4:
    st.subheader("4. 機構動向")
    st.metric(label="2800.HK 派發日 (25日內)", value=f"{dist_days} 日", delta="跌>0.2%且成交增加", delta_color="off")
    st.markdown(f"**狀態：{light_inst} ({score_inst}/2分)**")
    st.caption(desc_inst)

# --- 走勢參考圖表 ---
st.divider()
chart1, chart2 = st.columns(2)

with chart1:
    st.subheader("恒生指數近 1 年走勢與移動平均線")
    hsi_chart = hsi[["Close", "MA50", "MA200"]].tail(250).rename(
        columns={"Close": "恒指收市", "MA50": "50天線", "MA200": "200天線"}
    )
    st.line_chart(hsi_chart)

with chart2:
    st.subheader("債市避險比率 (HYG / IEF) 近 1 年走勢")
    bond_chart = bond_df[["Ratio", "MA20", "MA50"]].tail(250).rename(
        columns={"Ratio": "HYG/IEF 比率", "MA20": "20天線", "MA50": "50天線"}
    )
    st.line_chart(bond_chart)
