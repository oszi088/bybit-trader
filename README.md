# Kripto Trader AI Ügynök — Bybit kiadás (top 20, F&G, optimizer)

Súlyozott szavazós kereskedési ügynök 13 indikátorral és makro hangulattal, **Bybit spot** piacra hangolva, élesedett kockázatkezeléssel és portfólió-szintű multi-coin kezeléssel.

## Új képességek ebben a kiadásban

**Sub-5min granularitas (scalping mod).** Az ugynok mostmar **1m, 3m es 5m** timeframe-eken is fut, gyorsabb indikator periodusokkal (RSI=7, MACD=5/13/4, BB=14, ATR=7), szukebb ATR stopokkal (1.5×/2.5× a 3×/5× helyett), bekapcsolt trailing stoppal, magasabb threshold-dal (±0.55 a koltseg miatt), es lefele bovitett MTF cascade-del (5m → 15m → 1h → 4h sulyozottan). A `--scalping` flag-gel barmelyik parancs scalping-szeru hangolasra valt, vagy automatikusan bekapcsol ha `--timeframe 1m/3m/5m`. A poll periodus is automatikusan le van skalazva (1m → 10s, 3m → 20s, 5m → 30s).



**Top 20 USDT pár párhuzamosan.** A `portfolio` parancs egyszerre keres jelet a 20 legnagyobb USDT spot párra (BTC, ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, TRX, DOT, LINK, MATIC, TON, SHIB, LTC, BCH, NEAR, UNI, XLM, ATOM), egy közös cash poolban és **közös RiskManagerrel**. A `--max-positions` flag-gel limitálható a párhuzamosan nyitott pozíciók száma (alapból 5), így a korreláció miatti dupla kockázat el van kerülve. A `--symbols top5` vagy egyedi vesszős lista is megadható.

**Fear & Greed Index becsatolása.** Az alternative.me publikus API-jából 1 órás cache-sel olvassuk a napi makro hangulati indexet (0–100). Ez egy plusz **kontrarian indikátor** lett a szavazásban: extrém félelemnél (≤24) BUY szavazatot ad, extrém kapzsiságnál (≥75) SELL-t. Ha az API nem elérhető, biztonsági fallback `50` (neutral). Az aktuális érték közvetlenül lekérhető: `python main.py fg`.

**Multi-TimeFrame megerosites.** Az alacsonyabb timeframe-en (pl. 1h) hozott dontest 5 magasabb timeframe trendje is befolyasolja: 6h, 12h, 1d (napi), 1w (heti), 1M (havi). A "yearly" iranyt a havi gyertyak SMA-jabol szarmaztatjuk. Harom mod kozott valthatsz a `MTFConfig.mode`-dal: `weighted` (a magasabb tf-ek osszesitett trend score-ja egy plusz szavazokent szerepel), `gate` (ha az MTF erosen ellentmond a foi jelnek, a Decision-t HOLD-ra korlatozza), `off` (kikapcsolva). A backtest a CSV-bol resample-lel automatikusan eloallitja a magasabb tf-eket; eles modban CCXT-vel kerheto le. A hosszabb timeframe-ek nagyobb sulyt kapnak (1M = 2.0, 1w = 1.5, 1d = 1.0).

**Optimalizáló overfit-szűrővel.** Az `optimize` parancs az adatot 4 részre bontja (50% IS, 25% OOS1, 25% OOS2), és walk-forward grid search-csel 5 kulcsparamétert hangol (BUY/SELL küszöb + 3 fő indikátor súly). **Csak akkor fogadja el** az új paramétereket, ha mindkét OOS szakaszon nem-negatív hozam jön ki — így nem fognak átmenni az „IS-en csodás, élesben katasztrófa" kombinációk. A kereső tér szándékosan kicsi (3³ × 3 × 3 = 243 kombináció), hogy ne legyen érdemi tér a túlillesztésre.

## Üzemmódok

| Mód                 | Mit csinál                                              |
|---------------------|---------------------------------------------------------|
| `backtest`          | CSV szimuláció slippage-dzsel, ATR stops-okkal          |
| `walkforward`       | Több időablakon backteszt, stabilitás-mérés             |
| `optimize`          | Walk-forward grid search overfit-szűréssel              |
| `paper`             | Élő Bybit ár, szimulált broker, egy coin                |
| `live --testnet`    | Bybit testnet, valódi API útvonal                       |
| `live --live`       | Bybit.eu mainnet (alapból dry-run, `--execute` élesít)  |
| `portfolio paper`   | Top 20 coin szimulált broker, közös kockázat            |
| `portfolio live --testnet/--live` | Top 20 coin Bybit testnet vagy mainnet    |
| `decide`            | Egy döntés most a publikus piacról (F&G-vel)            |
| `trades`            | SQLite trade log listázás                               |
| `fg`                | Aktuális Fear & Greed Index lekérdezése                 |

## Indikátor készlet (13 szavazó)

| Csoport       | Indikátor                                   |
|---------------|---------------------------------------------|
| Trend         | SMA-cross, EMA-cross, MACD, ADX (+DI/-DI)   |
| Momentum      | RSI, Stochastic %K/%D, CCI                  |
| Volatilitás   | Bollinger Bands, ATR (méretezésre)          |
| Volumen       | OBV, VWAP, MFI                              |
| **Makro**     | **Fear & Greed Index** (kontrarian)         |

## Projekt struktúra

```
config.py        - paraméterek, súlyok (default/trend/range), kockázat, F&G config
coins.py         - top 20 / top 5 USDT spot pár lista + parser
indicators.py    - 12 indikátor pandas implementációja
signals.py       - indikátor → {-1, 0, +1} (F&G-vel)
fear_greed.py    - alternative.me API cache-elt lekérése
regime.py        - trend/range/neutral rezsim detektor
agent.py         - regime-aware súlyozott szavazás + F&G
data_source.py   - CSV + CCXT (Bybit testnet/eu/global)
brokers.py       - PaperBroker + BybitBroker
risk_manager.py  - plafon, napi/drawdown kill switch, vol-szűrő, Kelly
notify.py        - Telegram + log fallback
db.py            - SQLite trade log
trader.py        - élő/paper loop ATR stops, trailing, watchdog, db, notify
portfolio.py     - multi-symbol manager közös RiskManagerrel
optimizer.py     - walk-forward grid search overfit-szűréssel
backtest.py      - slippage, walk-forward, ATR stops
main.py          - CLI
```

## Telepítés

```bash
pip install -r requirements.txt
```

## API kulcsok

```bash
export BYBIT_API_KEY="..."
export BYBIT_API_SECRET="..."
# opcionalis ertesiteshez:
export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_CHAT_ID="..."
```

A Fear & Greed API **nem igényel kulcsot** (publikus végpont).

## Tipikus munkafolyamat

### 1) Optimalizálás történeti adaton

```bash
python main.py optimize --csv btc_1h_1y.csv --top 5
```

Az optimalizáló kiírja a default eredményét, majd a top 5 olyan paraméterkészletet, amely mindkét out-of-sample szakaszon nem-negatív. Ha **nincs ilyen** ("Nincs olyan paraméter-készlet…"), az nem hiba — azt jelenti, hogy az adat nem ad robusztus jelet, és **nem szabad** mainnetre menni.

### 2) Walk-forward stabilitás-mérés a választott paraméterrel

```bash
python main.py walkforward --csv btc_1h_1y.csv --fold-size 500
```

### 3) Aktuális Fear & Greed lekérdezése

```bash
python main.py fg
# Fear & Greed Index: 73 (Greed)
```

### 4) Egy döntés most

```bash
python main.py decide --symbol BTC/USDT --timeframe 4h --endpoint eu
```

A kimenet mutatja az F&G értéket és minden indikátor szavazatát.

### 5) Top 20 coin paper trading

```bash
python main.py portfolio paper --symbols top20 --max-positions 5 --max-drawdown 0.10
```

### 6) Top 5 coin Bybit testnet

```bash
python main.py portfolio live --testnet --symbols top5 --max-order 50
```

### 6.4) Historikus adat: Bybit vagy Binance

```bash
# Bybit (alapertelmezett, EU-konform): top 20, 1h, 1 ev:
python fetch_history.py --symbols top20 --timeframe 1h --years 1

# Binance: BTC 2017 augusztusatol (~9 ev), Bybit-en csak 2018 ota van:
python fetch_history.py --exchange binance --symbols BTC/USDT --timeframe 1d --years 9

# Binance teljes top 5, 1h+4h+1d, 5 evre vissza:
python fetch_history.py --exchange binance --symbols top5 \
    --timeframes 1h,4h,1d --years 5
```

A Binance-rol jovo CSV-k `_binance` szuffixet kapnak (pl. `BTC_USDT_1d_5y_binance.csv`), igy keverhetoek a Bybit-tel anelkul, hogy felulirnak. **Figyelem:**
- Binance EU/UK/US-bol blokkolt lehet — VPN-nel toltsd VPS-rol, ne a sajat IP-rol
- A Binance es Bybit arak kozott ~5-15 bps szpred van — egy Binance-on optimalizalt parameter-keszlet eles Bybit-en kis pertubaciot mutathat
- Mainnethez (Bybit.eu) erdemes Bybit-rol hatra teszteleni, hogy ne legyen exchange-specifikus mikrostruktura-bias

### 6.5) Scalping mod (sub-5min)

```bash
# 1m timeframe automatikusan scalping mod-ra valt:
python main.py paper --symbol BTC/USDT --timeframe 1m

# Explicit --scalping flaggel barmilyen tf-en:
python main.py decide --symbol ETH/USDT --timeframe 3m --endpoint eu --scalping

# Backteszt 1m CSV-vel:
python fetch_history.py --symbols BTC/USDT --timeframe 1m --years 0.25
python main.py backtest --csv data/BTC_USDT_1m_0.25y.csv --scalping --symbol BTC/USDT
```

**Figyelem scalping-rol:** Bybit spot fee 0.1%/oldal = 20 bps round-trip + ~7 bps slippage. Egy atlagos 1m mozgas csak 8-15 bps, igy a fee/slippage gyakran felemeszti a hozamot. Ajanlott scalpinghez:
- Bybit perpetual futures (taker 0.055%, maker -0.005% rebate)
- VIP fee szint (volumennel csokken a fee)
- Jol megvalasztott high-volatility ido (ne range-ben scalpolj)

A scalping mod tesztben (3 nap szintetikus 1m adat): 262 trade, 16% win rate, -26% — ez a kimenet azt mutatja, hogy szintetikus GBM-en (auto-correlation nelkul) a scalping nem mukodik. **Csak valos historikus 1m adattal lesz ertekes** a backteszt — a `fetch_history.py --timeframe 1m`-mel toltsd le.

### 7) Bybit.eu mainnet, top 5, dry-run (NEM küld valódi megbízást)

```bash
python main.py portfolio live --live --symbols top5 --max-order 25 --daily-loss 50
```

### 8) Mainnet éles megbízással (felelősséged tudatában)

```bash
python main.py portfolio live --live --execute --symbols top5 \
    --max-order 25 --daily-loss 50 --max-drawdown 0.08
```

## Védelmek

- **`--live` flag kötelező** valódi módhoz; alapból testnet
- **Megrendelésérték plafon** (`--max-order`)
- **Napi veszteséglimit** (`--daily-loss`)
- **Drawdown kill switch** (`--max-drawdown`)
- **Volatilitás-szűrő** (extrém ATR/ár arány felett nem nyit)
- **Kelly-arányos méret** (gyengébb score → kisebb pozíció)
- **Max párhuzamos pozíció** (portfolio módban)
- **Dry-run** valódi API kulccsal is csak logol
- **Interaktív `YES` megerősítés** mainneten

## Hangolás

A `config.py`-ban `StopConfig`, `RegimeConfig`, `RiskConfig`, `BacktestConfig`, `FearGreedConfig`, `WatchdogConfig`, és négy súlytábla (`DEFAULT_WEIGHTS`, `TREND_WEIGHTS`, `RANGE_WEIGHTS`, `SCALPING_WEIGHTS`) finomítható. Scalping presethez a `make_scalping_config(timeframe="1m")` factory hívható. Az `optimize` parancsmal kapott legjobb paramétereket kézzel írd be ide.

### Granularitas tablazat

| Timeframe | Poll period | Scalping mod | Indikator periodus | MTF cascade |
|-----------|-------------|--------------|--------------------|-------------|
| 1m        | 10s         | auto         | RSI=7, MACD=5/13/4 | 5m,15m,1h,4h |
| 3m        | 20s         | auto         | RSI=7, MACD=5/13/4 | 5m,15m,1h,4h |
| 5m        | 30s         | auto         | RSI=7, MACD=5/13/4 | 5m,15m,1h,4h |
| 15m       | 60s         | -            | RSI=14, MACD=12/26/9 | 6h,12h,1d,1w,1M |
| 1h        | 60s         | -            | RSI=14, MACD=12/26/9 | 6h,12h,1d,1w,1M |
| 4h        | 180s        | -            | RSI=14, MACD=12/26/9 | 6h,12h,1d,1w,1M |
| 1d        | 900s        | -            | RSI=14, MACD=12/26/9 | 6h,12h,1d,1w,1M |

## Felelősségvállalás

Ez oktatási kód, **nem pénzügyi tanácsadás**. Az automata kereskedés valódi pénzbeli veszteséget okozhat. Mielőtt mainnetre állsz: futtasd `optimize`-t éles BTC/ETH adaton, ellenőrizd `walkforward`-dal, futtasd 1-2 hetet `paper` és `testnet` módban, majd mainnet `dry-run`-t, és csak utána `--execute` kis `--max-order` mellett.
