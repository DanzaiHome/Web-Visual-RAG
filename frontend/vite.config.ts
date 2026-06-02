import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const apiTarget = 'http://127.0.0.1:8010';

export default defineConfig({
  base: './',
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    allowedHosts: ['nat2-notebook-inspire.sii.edu.cn'],
    watch: {
      usePolling: true,
      interval: 800,
      ignored: ['**/node_modules/**', '**/dist/**', '**/.git/**']
    },
    proxy: {
      '/api': apiTarget,
      '/health': apiTarget
    }
  },
  preview: {
    host: '0.0.0.0',
    port: 5173,
    allowedHosts: ['nat2-notebook-inspire.sii.edu.cn'],
    proxy: {
      '/api': apiTarget,
      '/health': apiTarget
    }
  }
});
