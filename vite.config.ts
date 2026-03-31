import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, process.cwd(), '');
    // VITE_ 로 시작하는 키를 우선적으로 찾습니다.
    const apiKey = env.VITE_GEMINI_API_KEY || env.GEMINI_API_KEY || "";

    return {
      server: { port: 3000, host: '0.0.0.0' },
      plugins: [react()],
      define: {
        // 모든 경로의 변수명을 지원하도록 강제 정의
        'process.env.VITE_GEMINI_API_KEY': JSON.stringify(apiKey),
        'process.env.GEMINI_API_KEY': JSON.stringify(apiKey),
        'process.env.API_KEY': JSON.stringify(apiKey),
      },
      resolve: {
        alias: { '@': path.resolve(__dirname, '.') }
      }
    };
});
