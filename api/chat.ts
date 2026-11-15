import { GoogleGenAI } from "@google/genai";

const SYSTEM = "You are Java, a helpful and friendly AI assistant. Reply in the same language as the user. Support Uzbek (Latin), English, and Russian. Be accurate, clear, practical, and honest when unsure. Format code with Markdown fences.";

type ChatMessage = { role: "user" | "assistant"; content: string };

export default async function handler(req: any, res: any) {
  res.setHeader("Content-Type", "application/json; charset=utf-8");

  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Faqat POST so'rovi qabul qilinadi." });
  }

  try {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return res.status(503).json({ error: "Gemini API kaliti sozlanmagan. Vercel Environment Variables bo'limiga GEMINI_API_KEY qo'shing." });
    }

    const body = req.body as { messages?: unknown } | undefined;
    if (!body || !Array.isArray(body.messages) || body.messages.length < 1 || body.messages.length > 40) {
      return res.status(400).json({ error: "Suhbat formati noto'g'ri." });
    }

    const contents = body.messages.map((item: unknown) => {
      if (!item || typeof item !== "object") throw new Error("Invalid message");
      const message = item as { role?: unknown; content?: unknown };
      if (!["user", "assistant"].includes(String(message.role)) || typeof message.content !== "string") {
        throw new Error("Invalid message");
      }
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
    if (!reply) return res.status(502).json({ error: "AI javobi bo'sh qaytdi. Qayta urinib ko'ring." });
    return res.status(200).json({ reply });
  } catch (error) {
    console.error("Java AI error:", error);
    return res.status(500).json({ error: "Java AI javob bera olmadi. API kalitini tekshiring." });
  }
}
