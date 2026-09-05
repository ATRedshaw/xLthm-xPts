import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        white: "rgb(var(--colour-white) / <alpha-value>)",
        black: "rgb(var(--colour-black) / <alpha-value>)",
        ink: {
          950: "rgb(var(--colour-ink-950) / <alpha-value>)",
          900: "rgb(var(--colour-ink-900) / <alpha-value>)",
          850: "rgb(var(--colour-ink-850) / <alpha-value>)",
          800: "rgb(var(--colour-ink-800) / <alpha-value>)",
          700: "rgb(var(--colour-ink-700) / <alpha-value>)",
        },
        signal: {
          300: "rgb(var(--colour-signal-300) / <alpha-value>)",
          350: "rgb(var(--colour-signal-350) / <alpha-value>)",
          450: "rgb(var(--colour-signal-450) / <alpha-value>)",
        },
        stone: {
          100: "rgb(var(--colour-stone-100) / <alpha-value>)",
          200: "rgb(var(--colour-stone-200) / <alpha-value>)",
          300: "rgb(var(--colour-stone-300) / <alpha-value>)",
          400: "rgb(var(--colour-stone-400) / <alpha-value>)",
          500: "rgb(var(--colour-stone-500) / <alpha-value>)",
          600: "rgb(var(--colour-stone-600) / <alpha-value>)",
          700: "rgb(var(--colour-stone-700) / <alpha-value>)",
        },
        amber: {
          100: "rgb(var(--colour-amber-100) / <alpha-value>)",
          200: "rgb(var(--colour-amber-200) / <alpha-value>)",
          300: "rgb(var(--colour-amber-300) / <alpha-value>)",
        },
        sky: {
          200: "rgb(var(--colour-sky-200) / <alpha-value>)",
          300: "rgb(var(--colour-sky-300) / <alpha-value>)",
        },
        rose: {
          200: "rgb(var(--colour-rose-200) / <alpha-value>)",
          300: "rgb(var(--colour-rose-300) / <alpha-value>)",
          400: "rgb(var(--colour-rose-400) / <alpha-value>)",
        },
        emerald: {
          400: "rgb(var(--colour-emerald-400) / <alpha-value>)",
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
