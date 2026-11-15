# Java AI

Java AI — Google Gemini bilan ishlaydigan AI yordamchi. Frontend Vite + React + TypeScript asosida, server API esa Netlify Functions orqali ishlaydi.

## Lokal ishga tushirish

1. Node.js 20+ o'rnating.
2. `npm install` bajaring.
3. `.env` fayl yarating va `GEMINI_API_KEY=...` qo'shing.
4. `npm run netlify:dev` bilan frontend va server funksiyani birga ishga tushiring.

Build: `npm run build`

Gemini API kalitini frontend kodiga yoki Git repository'ga joylamang. Netlify'da `GEMINI_API_KEY` ni Environment Variables bo'limida saqlang.
