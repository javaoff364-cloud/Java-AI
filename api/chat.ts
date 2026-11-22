// Javus AI inference endpoint. Python model serveri AI javoblarini yaratadi.
export default async function handler(req: any, res: any) {
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Faqat POST so'rovi qabul qilinadi." });
  }
  const baseUrl = process.env.JAVUS_API_URL;
  if (!baseUrl) return res.status(503).json({ error: "Javus Python model serveri ulanmagan. JAVUS_API_URL ni sozlang." });
  try {
    const response = await fetch(new URL("/chat", baseUrl), {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(req.body ?? {})
    });
    return res.status(response.status).json(await response.json());
  } catch (error) {
    console.error("Javus API proxy error:", error);
    return res.status(502).json({ error: "Javus model serveriga ulanib bo'lmadi." });
  }
}
