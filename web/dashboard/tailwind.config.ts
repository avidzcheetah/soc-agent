import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#f8fafc", // Clean light slate background
        card: "#ffffff",       // Pure white card
        "card-hover": "#f1f5f9",
        border: "#e2e8f0",     // Clean light border
        "border-subtle": "#f1f5f9",
        teal: {
          400: "#2dd4bf",
          500: "#0d9488",
          600: "#0f766e",
          700: "#115e59",
        },
        navy: {
          800: "#1e293b",
          900: "#0f172a",
          950: "#020617",
        }
      },
    },
  },
  plugins: [],
};
export default config;
