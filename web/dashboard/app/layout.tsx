import type { Metadata } from "next";
import "./globals.css";
import { TopBar } from "@/components/layout/TopBar";

export const metadata: Metadata = {
  title: "CORTEX-R | Cybersecurity Orchestration and Response through Transformer-guided EXecution and Reinforcement",
  description: "Optimizing Cybersecurity Incident Response Actions Using a Transformer-Based Agent.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-slate-50 text-slate-900 min-h-screen flex flex-col antialiased selection:bg-teal-500/20 selection:text-teal-900">
        <TopBar />
        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-7xl w-full mx-auto">
          {children}
        </main>
      </body>
    </html>
  );
}
