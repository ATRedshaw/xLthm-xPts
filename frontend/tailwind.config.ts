import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#09080d",
          900: "#0e0c13",
          850: "#121019",
          800: "#191621",
          700: "#24202e",
        },
        signal: {
          300: "#a5efc6",
          350: "#7ce3ad",
          450: "#4dcc88",
        },
      },
      fontFamily: {
        sans: ["Inter Variable", "Inter", "sans-serif"],
        mono: ["JetBrains Mono Variable", "JetBrains Mono", "monospace"],
      },
      boxShadow: {
        drawer: "-24px 0 64px rgba(0, 0, 0, 0.36)",
      },
    },
  },
  plugins: [],
} satisfies Config;
