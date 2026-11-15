import { GoogleGenAI } from "@google/genai";

const SYSTEM = "You are Java, a helpful and friendly AI assistant. Reply in the same language as the user. Support Uzbek (Latin), English, and Russian. Be accurate, clear, practical, and honest when unsure. Format code with Markdown fences.";

export default async (request: Request): Promise<Response> => {
  const json = (body: unknown, status = 200) =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json; charset=utf-8" } });

  if (request.method !== "POST") return json({ error: "Faqat POST so'rovi qabul qilinadi." }, 405);

  try {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) return json({ error: "Gemini API kaliti sozlanmagan. Netlify Environment Variables bo'limiga GEMINI_API_KEY qo'shing." }, 503);

    const body = await request.json() as { messages?: unknown };
    if (!Array.isArray(body.messages) || body.messages.length < 1 || body.messages.length > 40) {
      return json({ error: "Suhbat formati noto'g'ri." }, 400);
    }

    const contents = body.messages.map((item: unknown) => {
      if (!item || typeof item !== "object") throw new Error("Invalid message");
      const message = item as { role?: unknown; content?: unknown };
      if (!["user", "assistant"].includes(String(message.role)) || typeof message.content !== "string") throw new Error("Invalid message");
      return {
        role: message.role === "assistant" ? "model" : "user",
        parts: [{ text: message.content.slice(0, 12000) }]
      };
    });

    const ai = new GoogleGenAI({ apiKey });
    const result = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents,
      config: { systemInstruction: SYSTEM, maxOutputTokens: 4096, temperature: 0.7 }
    });
    const reply = result.text?.trim();
    if (!reply) return json({ error: "AI javobi bo'sh qaytdi. Qayta urinib ko'ring." }, 502);
    return json({ reply });
  } catch (error) {
    console.error("Java AI error:", error);
    return json({ error: "Java AI javob bera olmadi. API kalitini tekshiring." }, 500);
  }
};
