// Netlify proxy to the self-hosted Javus Python inference server.
export default async (request: Request): Promise<Response> => {
  const json = (body: unknown, status = 200) =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json; charset=utf-8" } });
  if (request.method !== "POST") return json({ error: "Faqat POST so'rovi qabul qilinadi." }, 405);
  const baseUrl = process.env.JAVUS_API_URL;
  if (!baseUrl) return json({ error: "Javus model serveri ulanmagan. JAVUS_API_URL sozlanishi kerak." }, 503);
  try {
    const body = await request.json();
    const upstream = await fetch(new URL("/chat", baseUrl), {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body)
    });
    return json(await upstream.json(), upstream.status);
  } catch (error) {
    console.error("Javus API proxy error:", error);
    return json({ error: "Javus model serveriga ulanib bo'lmadi." }, 502);
  }
};
