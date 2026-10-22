import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/layout/Sidebar";
import { TopBar } from "@/components/layout/TopBar";

export const metadata: Metadata = {
  title: "SOC Simulation Environment | Transformer-Based Incident Response Platform",
  description:
    "Visualization, monitoring, investigation, and evaluation interface for the Autonomous SOC Simulation Research Environment.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#080d1a] text-slate-100 min-h-screen flex antialiased selection:bg-teal-500/30 selection:text-teal-200">
        <Sidebar />
        <div className="flex-1 flex flex-col min-w-0">
          <TopBar />
          <main className="flex-1 p-6 overflow-y-auto max-w-7xl w-full mx-auto">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
