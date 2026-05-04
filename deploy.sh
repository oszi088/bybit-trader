#!/bin/bash
# =============================================================================
# Projekt feltöltése a VPS-re és tanítás indítása
#
# Futtatás Windows PowerShell-ből vagy Git Bash-ből:
#   bash deploy.sh <VPS_IP>
#
# Példa:
#   bash deploy.sh 95.179.200.100
# =============================================================================

VPS_IP="${1:?Használat: bash deploy.sh <VPS_IP>}"
VPS_USER="root"
REMOTE_DIR="/root/bybit-trader"

echo "======================================="
echo " Deploy → $VPS_USER@$VPS_IP"
echo "======================================="

# 1. Projekt feltöltése (kizárjuk a nagy/felesleges fájlokat)
echo "[1/3] Fájlok feltöltése..."
rsync -avz --progress \
    --exclude "__pycache__" \
    --exclude "*.pyc" \
    --exclude ".env" \
    --exclude "data/*.csv" \
    --exclude "models/*.pkl" \
    --exclude "trader_state.db*" \
    --exclude "*.db" \
    ./ "$VPS_USER@$VPS_IP:$REMOTE_DIR/"

# 2. Setup futtatása (csak első alkalommal szükséges)
echo ""
echo "[2/3] VPS setup..."
ssh "$VPS_USER@$VPS_IP" "bash $REMOTE_DIR/setup_vps.sh"

# 3. Training indítása screen-ben (háttérben fut, SSH bezárható)
echo ""
echo "[3/3] Training indítása háttérben..."
ssh "$VPS_USER@$VPS_IP" \
    "screen -dmS training bash -c \
    'cd $REMOTE_DIR && bash train_pipeline.sh 2>&1 | tee logs/trainer.log'"

echo ""
echo "======================================="
echo " Kész! A tanítás háttérben fut."
echo "======================================="
echo ""
echo " Visszacsatlakozás a VPS-re:"
echo "   ssh $VPS_USER@$VPS_IP"
echo "   screen -r training"
echo ""
echo " Log követés:"
echo "   ssh $VPS_USER@$VPS_IP 'tail -f $REMOTE_DIR/logs/trainer.log'"
echo ""
echo " Modellek letöltése (ha kész):"
echo "   scp -r $VPS_USER@$VPS_IP:$REMOTE_DIR/models ./models"
