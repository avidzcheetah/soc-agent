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
        background: "#080d1a", // Deep navy background
        card: "#0d1527",       // Slate navy panel
        "card-hover": "#131e36",
        border: "#1e293b",     // Subtle slate border
        "border-subtle": "#162033",
        teal: {
          400: "#2dd4bf",
          500: "#14b8a6",
          600: "#0d9488",
        },
        navy: {
          800: "#0f172a",
          900: "#080d1a",
          950: "#040711",
        }
      },
    },
  },
  plugins: [],
};
export default config;
