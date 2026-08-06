# data_collector.py
import feedparser, requests, yfinance as yf, pandas as pd, numpy as np
import time
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from dateutil import parser as dtparser
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pytz  # ✅ Required for timezone conversion

# ------------------ CONFIG ------------------
INDEX_TICKER = "PSEI.PS"
GLOBAL_START = "2021-01-01"
GLOBAL_END   = "2025-04-30"
RETURN_THRESH = 0.005

NEWSDATA_KEY = "pub_66d865c190554b4e9d856678ac34d6b6"
NEWSDATA_URL = "https://newsdata.io/api/1/news"

QUERIES = [
    "Philippines stock market", "PSE Composite Index", "Philippine economy",
    "Bangko Sentral ng Pilipinas policy", "Philippine inflation",
    "Philippine GDP data", "Philippines politics economy",
    "Philippine central bank", "Philippines interest rate decision",
    "Philippines election impact economy", "Philippines financial market",
    "Philippines budget announcement", "Philippine fiscal policy",
    "Philippine trade balance", "Philippines foreign investment"
]

RSS_TMPL = "https://news.google.com/rss/search?q={query}"
MAX_RSS = 1000
SLEEP_RSS = 1.0
SLEEP_MO = 2.0

# ---------- FinBERT ---------------------
tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
device = 0
sent = pipeline("text-classification", model=mdl, tokenizer=tok, top_k=None, device=device)

# Philippine timezone
ph_tz = pytz.timezone("Asia/Manila")

def text_sentiment(txt):
    res = {d["label"].lower(): d["score"] for d in sent(txt[:512])[0]}
    pos, neg = res.get("positive",0), res.get("negative",0)
    score = pos - neg
    conf = max(res.values()) if res else 0.0
    if score > 0.10: d = "Bullish"
    elif score < -0.10: d = "Bearish"
    else: d = "Neutral"
    return d, round(conf,4), round(score,4)

def fetch_month_close(tic, start, end):
    h = tic.history(start=start, end=end)
    if h.empty: return pd.Series(dtype=float)
    s = h["Close"].copy()
    s.index = s.index.date
    s.name = "Close"
    return s

def fetch_rss(q, start_dt, end_dt):
    filt = f" after:{start_dt:%Y-%m-%d} before:{end_dt:%Y-%m-%d}"
    url  = RSS_TMPL.format(query=f"{q}{filt}".replace(" ", "%20"))
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:MAX_RSS]:
        title = e.get("title","").strip()
        pp = e.get("published_parsed") or e.get("updated_parsed")
        if not title or not pp: continue
        dt = datetime(*pp[:6], tzinfo=timezone.utc)
        if start_dt.date() <= dt.date() < end_dt.date():
            items.append({"text": title, "date": dt, "source": "RSS"})
    return items

def fetch_newsdata(q, start_dt, end_dt):
    params = {
        "apikey": NEWSDATA_KEY, "q": q, "country": "ph", "language": "en",
        "from_date": f"{start_dt:%Y-%m-%d}", "to_date": f"{end_dt:%Y-%m-%d}",
        "page": 1
    }
    all_items = []
    while True:
        try:
            r = requests.get(NEWSDATA_URL, params=params, timeout=30)
            js = r.json()
            for art in js.get("results", []):
                if not isinstance(art, dict): continue
                title = art.get("title", "").strip()
                pub = art.get("pubDate")
                if not title or not pub: continue
                try:
                    dt = dtparser.isoparse(pub)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                if start_dt.date() <= dt.date() < end_dt.date():
                    all_items.append({"text": title, "date": dt, "source": "Newsdata"})
            if not js.get("nextPage"): break
            params["page"] += 1
            time.sleep(0.2)
        except:
            break
    return all_items

def get_prev_next(closes, dt):
    d = dt.date()
    idx = closes.index
    if len(idx)==0 or d<idx[0] or d>=idx[-1]: return None, None
    nxt = idx[idx > d]
    prv = idx[idx <= d]
    if len(nxt)==0 or len(prv)==0: return None, None
    return closes[prv[-1]], closes[nxt[0]]

def run_data_collection():
    tic = yf.Ticker(INDEX_TICKER)
    start_dt = datetime.fromisoformat(GLOBAL_START)
    end_dt   = datetime.fromisoformat(GLOBAL_END)

    rows = []
    mo = start_dt.replace(day=1)
    while mo < end_dt:
        nxt_mo = mo + relativedelta(months=1)
        closes = fetch_month_close(tic, mo, nxt_mo)
        if closes.empty: mo = nxt_mo; continue

        all_news = []
        for q in QUERIES:
            all_news += fetch_rss(q, mo, nxt_mo)
            all_news += fetch_newsdata(q, mo, nxt_mo)
            time.sleep(SLEEP_RSS)

        for r in pd.DataFrame(all_news).drop_duplicates(["text", "date"]).to_dict("records"):
            prev_c, next_c = get_prev_next(closes, r["date"])
            if prev_c is None: continue
            pct = (next_c - prev_c) / prev_c
            price_dir = (
                "Bullish" if pct > RETURN_THRESH else
                "Bearish" if pct < -RETURN_THRESH else "Neutral"
            )
            t_dir, t_conf, t_score = text_sentiment(r["text"])
            txt_len = len(r["text"])
            has_excl = int("!" in r["text"])
            has_ques = int("?" in r["text"])

            # ✅ Convert to PH time and flag headlines after 3 PM
            date_ph = r["date"].astimezone(ph_tz)
            post_3pm = int(date_ph.hour >= 15)

            rows.append({
                **r,
                "prev_close": prev_c,
                "next_close": next_c,
                "pct_ret": pct,
                "price_dir": price_dir,
                "text_dir": t_dir,
                "text_conf": t_conf,
                "sentiment_score": t_score,
                "txt_len": txt_len,
                "has_excl": has_excl,
                "has_ques": has_ques,
                "post_3pm_ph": post_3pm  # ✅ Added to CSV
            })

        mo = nxt_mo

    df = pd.DataFrame(rows).drop_duplicates(["text", "date"])
    df.to_csv("psei_labeled_news.csv", index=False)
    print("✅ Data saved to psei_labeled_news.csv")

if __name__ == "__main__":
    run_data_collection()
