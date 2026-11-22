# Javus AI

**Javus AI** — o'z arxitekturasi, o'zbekcha datasetlari va o'zida o'qitiladigan neyron tarmog'iga ega mustaqil multimodal sun'iy intellekt loyihasi.

Birinchi modelimiz: **Javus Alpha**. Hozirgi holati — kichik, faqat matn bilan ishlaydigan eksperimental Transformer prototipi. Bu hali tayyor yoki kuchli chatbot emas.

## Texnologiyalar
- Python 3.10+ va PyTorch
- 256 tokenli UTF-8 byte tokenizer
- Decoder-only Transformer, causal self-attention
- AdamW, cross-entropy, gradient clipping
- Smart Training, validation loss monitoring va adaptive learning rate
- pytest va GitHub Actions

## O'rnatish

Termux ichida Ubuntu proot muhitidan foydalanish tavsiya etiladi:

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

## Testlar

```bash
python -m pytest -q
```

## Javus Alpha treningi

Bir martalik 100 qadam:

```bash
python -m javus.train --steps 100
```

Har 100 qadamdan keyin checkpoint saqlab, uzluksiz davom etish:

```bash
python -m javus.train --steps 100 --batch-size 8 --forever
```

Mavjud checkpointni chetlab, noldan boshlash uchun `--fresh` qo'shing. Checkpoint `checkpoints/javus-small.pt` da saqlanadi (eski trening fayli bilan moslik uchun nom saqlab qolingan). Smart Training natijalari `checkpoints/javus-alpha-metrics.jsonl` ga yoziladi.

## Smart Training nimalar qiladi?
1. Har bir 100 qadamlik batch yakunida modelni baholaydi.
2. Trening va validation loss ko'rsatkichlarini qayd etadi.
3. Validation loss yaxshilanmay qolsa, uchta baholashdan keyin learning rate'ni 50% kamaytiradi.
4. Model, optimizer, global step va Smart Training holatini checkpoint'ga saqlaydi.
5. Keyingi ishga tushishda oldingi holatdan davom etadi.

Validation uchun datasetning 25% qismi ajratiladi. Dataset juda kichik bo'lgani sababli hozirgi metrikalar faqat pipeline sinovi hisoblanadi; sifatli o'zbekcha korpus kerak.

## GitHub Actions
Training workflow har 6 soatda rejalashtirilgan va qo'lda ham ishga tushirilishi mumkin. GitHub-hosted runner doimiy server emas: har bir ishga tushish vaqt limiti bilan cheklangan. Checkpoint Actions cache orqali keyingi ishga tushishga uzatiladi, artifact esa 14 kun saqlanadi. Bu doimiy va kafolatlangan model ombori o'rnini bosmaydi.

## Matn generatsiyasi

```bash
python -m javus.generate --prompt "Assalomu alaykum" --max-new-tokens 40
```

## Keyingi bosqichlar
- [x] Python Transformer prototipi
- [x] Dastlabki core testlar
- [x] Javus Alpha nomi va Smart Training
- [ ] Keng, toza va litsenziyasi tekshirilgan o'zbekcha dataset
- [ ] Kengaytirilgan baholash va model benchmarklari
- [ ] Tokenizer va arxitekturani takomillashtirish
- [ ] Vision va audio modullari
- [ ] Javus AI 1.0

## Litsenziya
Apache-2.0
