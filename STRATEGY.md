# A kereskedési stratégia magyarázata

Ez a dokumentum elmagyarázza, hogyan működik a rendszer — mitől dönti el a bot, hogy vegyen, adjon el, vagy ne csináljon semmit.

---

## 1. Az egész rendszer egy mondatban

> Minden gyertyánál 15 indikátor szavaz (+1 / 0 / −1), a szavazatokat súlyozzuk, az összesített score-t összehasonlítjuk egy küszöbértékkel — de mielőtt ténylegesen belép, több szűrőrétegen is át kell jutnia a jelzésnek.

---

## 2. Az indikátorok és szavazataik

Az indikátorok mindegyike **egy gyertyára** ad véleményt: `+1` (vegyél), `0` (semleges), `−1` (adj el).

### Trend indikátorok
| Indikátor | Mit mér | Mikor vételi (+1) | Mikor eladási (−1) |
|-----------|---------|-------------------|---------------------|
| **SMA cross** | Gyors (20) vs. lassú (50) mozgóátlag | gyors > lassú | gyors < lassú |
| **EMA cross** | Ugyanaz, de exponenciális (gyorsabb) | gyors EMA > lassú EMA | gyors EMA < lassú EMA |
| **MACD** | Trend lendület (12/26/9 EMA különbség) | hisztogram > 0 | hisztogram < 0 |
| **ADX** | Trend erőssége + iránya | ADX > 20 és +DI > −DI | ADX > 20 és −DI > +DI |
| **Golden/Death cross** | 50MA vs. 200MA (hosszú trend) | 50MA > 200MA | 50MA < 200MA |
| **Long trend** | Ár a 200MA felett van-e | ár > 200MA | ár < 200MA |

### Momentum indikátorok
| Indikátor | Mit mér | Vételi | Eladási |
|-----------|---------|--------|---------|
| **RSI** | Túladott/túlvett állapot (14 periódus) | RSI < 30 (túladott) | RSI > 70 (túlvett) |
| **Stochastic** | RSI-hoz hasonló, de %K/%D keresztezéssel | %K < 20 és %K > %D | %K > 80 és %K < %D |
| **CCI** | Ár eltérése az átlagtól | CCI < −100 | CCI > +100 |

### Volatilitás
| Indikátor | Mit mér | Vételi | Eladási |
|-----------|---------|--------|---------|
| **Bollinger** | Ár a sávhoz képest (mean reversion) | ár < alsó sáv | ár > felső sáv |
| **ATR** | Volatilitás mértéke | *nem ad irányt* — szűrőként használják | — |

### Volumen indikátorok
| Indikátor | Mit mér | Vételi | Eladási |
|-----------|---------|--------|---------|
| **OBV** | Kumulált volumen iránya | OBV nő | OBV csökken |
| **VWAP** | Volumennel súlyozott átlagár | ár > VWAP | ár < VWAP |
| **MFI** | "Pénzbeáramlás" indikátor | MFI < 20 | MFI > 80 |
| **Fear & Greed** | Piaci hangulat (0–100) | < 20 (extreme fear) | > 80 (extreme greed) |

---

## 3. A szavazatokból score

Az indikátorok szavazatait **nem egyenlő súllyal** vesszük figyelembe:

```
score = Σ(súly × szavazat) / Σ(súlyok)
```

Például ha MACD súlya 1.5 és +1-et szavaz, RSI súlya 1.5 és +1-et szavaz,
de Bollinger súlya 1.0 és −1-et szavaz:

```
score = (1.5×1 + 1.5×1 + 1.0×(−1)) / (1.5 + 1.5 + 1.0)
      = (1.5 + 1.5 − 1.0) / 4.0
      = 2.0 / 4.0 = +0.50
```

A score mindig **−1.0 és +1.0 között** van.

### Default súlyok

```
MACD:         1.5   (trend, erős szignál)
RSI:          1.5   (momentum, erős szignál)
SMA cross:    1.0
EMA cross:    1.0
Bollinger:    1.0
Golden cross: 1.5
OBV:          0.7
VWAP:         0.8
Fear & Greed: 0.8
Stochastic:   1.0
Long trend:   0.7
CCI:          0.5
ADX:          0.5
MFI:          0.5
ATR:          0.5   (nem szavaz, de bent van a rendszerben)
```

### A küszöb

```
score ≥  0.40  →  BUY szándék
score ≤ −0.40  →  SELL szándék
egyébként       →  HOLD
```

---

## 4. A rezsim detektor (ADX-alapú)

A piacon két alapállapot van:

- **Trend**: az ár egyértelműen emelkedik vagy esik
- **Range (oldalazás)**: az ár sávban mozog, visszatér az átlaghoz

A bot az ADX értékéből állapítja meg:

| ADX érték | Rezsim | Súlykészlet |
|-----------|--------|-------------|
| > 25 | **Trend** | MACD, SMA/EMA cross, ADX, OBV, VWAP hangsúlyos |
| < 18 | **Range** | RSI, Stochastic, CCI, Bollinger, MFI hangsúlyos |
| 18–25 | **Neutral** | Default súlyok |

**Miért fontos?** Trendben a mean-reversion indikátorok (RSI, Bollinger) hamis jelzést adnak — "minden túlvett" miközben az ár egyre feljebb megy. Range-ben a trendkövetők hamis jelzést adnak. A rezsim detektor ezt kezeli.

---

## 5. A piaci ciklus detektor

Ez egy magasabb szintű szűrő, amely a **teljes piaci fázist** azonosítja — nem egy gyertya szintjén, hanem hetek/hónapok skáláján.

### A 9 ciklus

| Ciklus | Mit jelent | Long megengedett? |
|--------|-----------|------------------|
| `accumulation` | Alap formáció, okos pénz vásárol csendben | ✅ igen |
| `bull_early` | Kitörés a 200MA fölé, BTC vezet | ✅ igen |
| `bull_mid` | Erős uptrend, altcoin szezon indul | ✅ igen |
| `bull_late` | Parabolikus, extrém greed, bármikor fordulhat | ✅ igen (kis pozíció!) |
| `distribution` | Topp formáció, nagy kezek adnak el | ⚠️ csak magas meggyőzésre |
| `bear_early` | Gyors esés, pánik | ❌ **nem** |
| `bear_mid` | Lassú grinding, érdektelenség | ❌ **nem** |
| `altseason` | BTC dominancia zuhan, alts szárnyalnak | ✅ igen (altok preferálva) |
| `risk_off` | Makro sokk (VIX spike, szabályozói hír) | ❌ **nem** |

### Hogyan detektálja?

7 mutatót számít 0–1 skálán:
- Ár helyzete a 200MA-hoz képest
- Rövid távú momentum (20 napos hozam)
- Volatilitás szintje
- RSI értéke
- Fear & Greed index
- Funding rate szintje
- Bitcoin halving cikluson belüli pozíció (~4 éves ciklus)

Minden ciklushoz van egy "ideális profil" — a legjobban illeszkedőt választja.

### Hatása a döntésre

Ha a ciklus `bear_early`, akkor **még ha a score +0.80 is, a BUY-t letiltja**. Ez magyarázza, hogy az ETH adaton 0 trade volt — a cycle detektor bearish ciklust látott, és minden long szándékot blokkolt.

Emellett ciklusonként más küszöb és pozícióméret él:
- `bull_mid`-ban: threshold −0.03 (könnyebben vesz), pozíció 25%
- `bull_late`-ben: threshold +0.05 (nehezebben vesz), pozíció csak 10%

---

## 6. A döntési folyamat teljes sorrendben

Minden gyertyánál ez történik:

```
1. Indikátorok számítása (compute_all)
        ↓
2. Rezsim detektálás (ADX alapján: trend/range/neutral → súlyok kiválasztása)
        ↓
3. 15 indikátor szavaz → súlyozott score [-1.0, +1.0]
        ↓
4. Score ≥ threshold? → BUY szándék
        ↓
5. Timing szűrő (pl. hétvégén alacsonyabb aktivitás)
        ↓
6. Ciklus szűrő (allow_long? score ≥ ciklus-threshold?)
        ↓
7. Altseason validáció (ha altseason: megfelelő coin-e?)
        ↓
8. ML meta-label (opcionális: XGBoost szűrő, elég magas a valószínűség?)
        ↓
9. Override motor (ha blokkolt volt: elég erős-e az evidence az override-hoz?)
        ↓
10. Végső döntés: BUY / SELL / HOLD
```

Az **ATR volatilitás szűrő** a backtesztben is él: ha `ATR/ár ≥ 5%`, nem lépünk be (túl kockázatos belépési pont).

---

## 7. Pozícióméretezés

Ha a döntés BUY, a pozíció méretét két dolog befolyásolja:

**Score-arányos méretezés (Kelly-szerű):**
```
tényleges méret = alap_méret × |score|
```
Ha a score pontosan 0.40 (épp átlépte a küszöböt), kisebb pozíciót nyit.
Ha a score 0.90 (erős meggyőzés), közel teljes méretű pozíciót nyit.

**Stop-loss és take-profit (ATR-alapú):**
```
stop_loss  = belépési ár − 3.0 × ATR
take_profit = belépési ár + 5.0 × ATR
```
Az ATR (Average True Range) a piac volatilitásától függően változik — így viharos piacban távolabb van a stop, csendes piacban közelebb.

---

## 8. A backteszt logika

A backteszt a **történeti adaton szimulálja a kereskedést** — mintha valódi pénzzel kereskedtünk volna.

### Minden gyertyán:

1. **Trailing stop frissítés** — ha a pozíció van nyitva, és az ár feljebb ment, a stop is feljebb húzódik
2. **SL/TP ellenőrzés** — ha a gyertya low/high elérte a stop/TP szintet, zárjuk
3. **Ügynök döntése** — az indikátorok alapján BUY/SELL/HOLD
4. **Belépés/kilépés** — SL/TP után ugyanazon a gyertyán nem nyitunk újra
5. **Equity számítás** — cash + nyitott pozíció mark-to-market értéke

### Realisztikus elemek:
- **Slippage**: a fill ár 1–3 bázisponttal rosszabb a piaci árnál
- **Spread**: vételnél dráguló, eladásnál olcsóbb fill ár
- **Fee**: 0.1% tranzakciós díj minden oldalon
- **Fill a gyertya close-on** történik (nem a következő nyitón — ez kis optimista torzítás)

---

## 9. A portfolió-backteszter

A sima backteszt egyetlen coinon fut, **elkülönített tőkéből**.

A portfolió-backteszter **megosztott tőkéből** fut egyszerre több coinon:

```
Induló tőke: $10 000
Max pozíciók: 3
Slot méret:  $10 000 / 3 = ~$3 333 / pozíció
```

Ha BTC és SOL is ad egyszerre BUY jelzést, és van 2 szabad slot, mindkettőbe belép. Ha már 3 pozíció van nyitva, nem nyit újat — megvárja míg valamelyik zárul.

### Különbség a sima backteszttől:

| | Sima backteszt | Portfolió-backteszt |
|--|---------------|---------------------|
| Tőke | Minden coin külön $10 000 | Közös $10 000 |
| Egyidejű pozíciók | 1 (csak BTC VAGY ETH) | max N (BTC ÉS ETH egyszerre) |
| Valóság | Nem realisztikus | Közelebb a valósághoz |
| Drawdown | Nincs korreláció | Ha 3 coin egyszerre esik, a DD addódik |

---

## 10. Az optimalizáló

Az optimalizáló megkeresi azt a paraméterkészletet, amely **nem csak a múlt adatán teljesít jól** (overfitting), hanem ismeretlen adatokon is.

### Walk-forward módszer:

```
Teljes adat: ████████████████████████████████
              [───── IS ─────][──OOS1──][─OOS2─]
                  (50%)          (25%)    (25%)
```

- **IS (in-sample)**: ezen keresi a legjobb paramétereket
- **OOS1 és OOS2**: ezen ellenőrzi, hogy a paraméter valóban működik-e

Egy paraméterkészlet csak akkor "robusztus", ha:
1. IS-en jobb az alapkonfigurációnál
2. OOS1-en is pozitív a hozam
3. OOS2-n is pozitív a hozam

---

## 11. Mi hiányzik még (tervezett)

- **Bear stratégia**: bearish ciklusban jelenleg a bot kiáll (hold). A terv: short pozíció nyitása bear ciklusban — az ár esésére is lehet kereskedni (csak perpetual/futures piacon).
- **Portfolió optimalizáló**: az optimalizáló jelenleg egy coinra fut; a cél, hogy a portfolió-backteszterrel kombinálva több coin paraméterkészletét egyszerre optimalizálja.
