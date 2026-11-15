import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "Java AI — Aqlli yordamchingiz",
  description: "Java AI — savollaringizga javob beradigan shaxsiy sun’iy intellekt yordamchisi.",
  icons: { icon: "/java-mark.svg", shortcut: "/java-mark.svg", apple: "/java-mark.svg" },
};
export default function RootLayout({ children }: Readonly<{children: React.ReactNode}>) { return <html lang="uz"><body>{children}</body></html>; }
