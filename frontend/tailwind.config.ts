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
        violet: {
          350: "#b69aff",
          450: "#9871f4",
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
