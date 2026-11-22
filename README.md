# Javus AI

**Javus AI** — o'z arxitekturasi, o'zbekcha datasetlari va o'zida o'qitilgan neyron tarmog'iga ega mustaqil multimodal sun'iy intellekt yaratish loyihasi.

Hozirgi bosqich: **Javus Alpha 0.1.0 — ilk text-only model**.

> Bu loyiha tayyor generativ AI API'siga ulanmaydi. Decoder-only Transformer arxitekturasi PyTorch yordamida loyihaning o'zida yozilgan. Dastlabki model kichik tajriba uchun; u hali bilimli yoki ishonchli chatbot emas.

## Hozirgi texnologiyalar
- Python 3.10+
- PyTorch
- 256 o'lchamli UTF-8 byte tokenizer
- Causal self-attention va decoder-only Transformer
- AdamW, cross-entropy va gradient clipping
- pytest va GitHub Actions

## Termux orqali ishga tushirish

PyTorch Android/Termux muhitida oddiy pip install torch bilan har doim ham o'rnatilmaydi. Tavsiya etilgan yo'l — Termux ichida Ubuntu proot muhitidan foydalanish.

Termux:
```bash
pkg update && pkg upgrade -y
pkg install -y git proot-distro
proot-distro install ubuntu
proot-distro login ubuntu
```

Ubuntu ichida:
```bash
apt update
apt install -y python3 python3-venv python3-pip git
git clone https://github.com/javaoff364-cloud/Java-AI.git
cd Java-AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Agar PyTorch o'rnatilishi qurilma arxitekturasi yoki Python versiyasi sababli xato bersa, xato matnini tekshirish kerak; mos kelmaydigan wheel'ni majburan o'rnatmang.

## Core testlari
```bash
python -m pytest -q
```

## Birinchi modelni o'qitish
```bash
python -m javus.train --steps 100
```
Dastlab 100 qadam bilan sinab ko'ring. Smart Training har partiyadan keyin train va validation loss'ni hisoblaydi, validation natijasi yomonlashganda learning rate'ni avtomatik pasaytiradi. Eng yaxshi checkpoint `checkpoints/javus-alpha-best.pt`, oxirgi checkpoint esa `checkpoints/javus-alpha.pt` faylida saqlanadi.

Uzluksiz Smart Training:
```bash
python -m javus.train --steps 100 --batch-size 8 --forever
```

## Matn generatsiyasi
```bash
python -m javus.generate --prompt "Assalomu alaykum" --max-new-tokens 40
```

## Model arxitekturasi
1. UTF-8 byte tokenizer (256 token)
2. Token va pozitsiya embeddinglari
3. Causal multi-head self-attention
4. Feed-forward bloklar, residual ulanishlar va LayerNorm
5. Next-token prediction

## Dataset
`data/train.txt` hozircha kichik, qo'lda yozilgan sinov korpusi. U faqat pipeline ishlashini tekshirishga xizmat qiladi; undan mazmunli chatbot kutib bo'lmaydi. Kengaytiriladigan korpus litsenziyasi mos, kelib chiqishi qayd etilgan, maxfiy bo'lmagan va sifat nazoratidan o'tgan o'zbekcha matnlardan tuziladi.

## Rivojlanish rejasi
- [x] Python asosidagi Javus Alpha mavjud
- [x] Core uchun dastlabki avtomatik testlar
- [ ] Termux qurilmasida PyTorch muhitini tasdiqlash
- [ ] Javus Alpha modelini to'liq o'qitish va checkpointni tekshirish
- [x] Smart Training: validation, adaptive learning rate va best checkpoint
- [ ] O'zbekcha datasetni sifatli kengaytirish
- [ ] Tokenizer va trening pipeline'ni takomillashtirish
- [ ] Vision va audio encoderlar bilan multimodal arxitektura
- [ ] Javus AI 1.0

## Litsenziya
Apache-2.0
