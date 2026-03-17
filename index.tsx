
import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { ErrorBoundary } from './components/ErrorBoundary';

let rootInstance: any = null;

const initApp = () => {
  const rootElement = document.getElementById('root');
  if (!rootElement || rootInstance) return;

  try {
    rootInstance = createRoot(rootElement);
    rootInstance.render(
      <React.StrictMode>
        <ErrorBoundary>
          <App />
        </ErrorBoundary>
      </React.StrictMode>
    );
    
    const splash = document.getElementById('splash-screen');
    if (splash) {
      splash.style.opacity = '0';
      setTimeout(() => { if(splash.parentNode) splash.remove(); }, 500);
    }
  } catch (err) {
    console.error("Mounting error:", err);
    const errorLog = document.getElementById('mobile-error-log');
    if (errorLog) {
      errorLog.style.display = 'block';
      errorLog.innerHTML += '<div>Mount Error: ' + String(err) + '</div>';
    }
  }
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
