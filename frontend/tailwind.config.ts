import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        shell: '#151718',
        panel: '#f7f8f5',
        ink: '#1d2521',
        success: '#1f7a4d',
        caution: '#b7791f',
        action: '#2563eb',
      },
      boxShadow: {
        surface: '0 1px 2px rgb(15 23 42 / 0.08)',
      },
    },
  },
  plugins: [],
} satisfies Config;