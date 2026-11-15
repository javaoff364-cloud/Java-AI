import { GoogleGenAI } from "@google/genai";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
const SYSTEM = "You are Java, a helpful and friendly AI assistant. Reply in the same language as the user. Support Uzbek (Latin), English, and Russian. Be accurate, clear, practical, and honest when unsure. Format code with Markdown fences.";

export async function POST(request: Request) {
  try {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) return NextResponse.json({ error: "Gemini API kaliti sozlanmagan. Vercel Environment Variables bo'limiga GEMINI_API_KEY qo'shing." }, { status: 503 });
    const body = await request.json();
    const messages = body.messages;
    if (!Array.isArray(messages) || messages.length < 1 || messages.length > 40) {
      return NextResponse.json({ error: "Suhbat formati noto'g'ri." }, { status: 400 });
    }
    const contents = messages.map((m: {role:string;content:string}) => {
      if (!["user","assistant"].includes(m.role) || typeof m.content !== "string") throw new Error("Invalid message");
      return { role: m.role === "assistant" ? "model" : "user", parts: [{ text: m.content.slice(0, 12000) }] };
    });
    const ai = new GoogleGenAI({ apiKey });
    const result = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents,
      config: { systemInstruction: SYSTEM, maxOutputTokens: 4096, temperature: 0.7 }
    });
    const reply = result.text?.trim();
    if (!reply) return NextResponse.json({ error: "AI javobi bo'sh qaytdi. Qayta urinib ko'ring." }, { status: 502 });
    return NextResponse.json({ reply });
  } catch (error) {
    console.error("Java AI error:", error);
    return NextResponse.json({ error: "Java AI javob bera olmadi. API kalitini tekshiring." }, { status: 500 });
  }
}
