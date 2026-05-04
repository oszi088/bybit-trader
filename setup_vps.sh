#!/bin/bash
# =============================================================================
# VPS egyszeri telepítő script — Ubuntu 22.04 LTS
# Futtatás: bash setup_vps.sh
# =============================================================================

set -e   # első hiba esetén megáll

echo "======================================"
echo " Bybit Trader — VPS Setup"
echo "======================================"

# --- Rendszer frissítés ---
echo "[1/6] Rendszer frissítés..."
apt-get update -qq
apt-get install -y -qq \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip \
    git curl wget screen htop \
    build-essential   # xgboost fordításhoz kell

# --- Python venv ---
echo "[2/6] Python virtual environment létrehozása..."
python3.11 -m venv /root/trader-env
source /root/trader-env/bin/activate

pip install --upgrade pip -q

# --- Projekt felmásolás ---
# A projektet SCP-vel kell feltölteni a VPS-re:
#   scp -r C:\Users\Oszi\PycharmProjects\bybit-trader root@<VPS_IP>:/root/bybit-trader
# Ez a script feltételezi hogy már ott van.

PROJECT_DIR="/root/bybit-trader"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "HIBA: $PROJECT_DIR nem létezik."
    echo "Töltsd fel a projektet:"
    echo "  scp -r ./bybit-trader root@<VPS_IP>:/root/"
    exit 1
fi

cd "$PROJECT_DIR"

# --- Dependencies ---
echo "[3/6] Python csomagok telepítése..."
pip install -r requirements.txt -q

# --- Könyvtárak ---
echo "[4/6] Könyvtárak létrehozása..."
mkdir -p data models logs

# --- Környezeti változók ---
echo "[5/6] .env fájl sablon létrehozása..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# Bybit API (opcionális — csak élő kereskedéshez kell)
export BYBIT_API_KEY=""
export BYBIT_API_SECRET=""

# Telegram értesítés (opcionális)
export TELEGRAM_BOT_TOKEN=""
export TELEGRAM_CHAT_ID=""
EOF
    echo "  → .env sablon létrehozva, töltsd ki az értékeket!"
fi

# --- screen alias ---
echo "[6/6] Hasznos aliasok beállítása..."
cat >> /root/.bashrc << 'EOF'

# Bybit Trader
alias trader-env="source /root/trader-env/bin/activate && cd /root/bybit-trader"
alias trader-log="tail -f /root/bybit-trader/logs/trainer.log"
EOF

echo ""
echo "======================================"
echo " Setup kész!"
echo "======================================"
echo ""
echo "Következő lépés:"
echo "  1. Töltsd ki a .env fájlt (Bybit API kulcsok)"
echo "  2. Futtasd: bash train_pipeline.sh"
echo ""
echo "VPS hasznos parancsok:"
echo "  screen -S training          — új screen session"
echo "  screen -r training          — visszacsatlakozás"
echo "  Ctrl+A, D                   — session háttérbe"
