# Javus AI

Javus AI — o'z neyron tarmog'imizni noldan yaratish va o'qitish loyihasi. Hozirgi bosqich: **Javus Core 0.1.0**.

> Muhim: ushbu versiya Gemini yoki boshqa tayyor generativ AI API'sidan foydalanmaydi. Model arxitekturasi PyTorch yordamida noldan yozilgan kichik decoder-only Transformer. Dastlabki natijalar tajriba darajasida; u hali umumiy bilimga ega chatbot emas.

## Talablar
- Python 3.10+
- PyTorch
- CPU'da kichik tajribalarni bajarish mumkin; tezroq o'qitish uchun CUDA GPU tavsiya etiladi.

## O'rnatish
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Birinchi modelni o'qitish
```bash
python -m javus.train --steps 500
```
Checkpoint `checkpoints/javus-small.pt` ichiga saqlanadi.

## Matn yaratish
```bash
python -m javus.generate --prompt "Assalomu alaykum" --max-new-tokens 80
```

## API server
```bash
uvicorn javus.server:app --host 0.0.0.0 --port 8000
```
Health: `/health`; chat: `POST /chat`.

## Arxitektura
- 256 tokenli UTF-8 byte tokenizer
- Token va positional embedding
- Causal self-attention
- Feed-forward, residual connection va LayerNorm
- Next-token prediction, cross-entropy loss va AdamW
- Checkpoint saqlash/yuklash

## Dataset
`data/train.txt` — pipeline sinovi uchun kichik demo. Haqiqiy sifatli model uchun katta, tozalangan, litsenziyasi mos va maxfiy bo'lmagan dataset kerak. Demo matn kuchli chatbot yaratishga yetarli emas.

## Keyingi bosqichlar
1. Tokenizer va training pipeline'ni yaxshilash.
2. O'zbek tilidagi datasetni kengaytirish va baholash.
3. Vision encoder, audio encoder va multimodal fusion.
4. Inference API va Javus web interfeysini ulash.

## Litsenziya
Apache-2.0
