import React, { useState, useEffect, useCallback } from 'react';
import { FileUpload } from './components/FileUpload';
import { AnalysisView } from './components/AnalysisView';
import { ChatInterface } from './components/ChatInterface';
import { PreMeetingView } from './components/PreMeetingView';
import { PersonaSettings } from './components/PersonaSettings';
import { analyzeSalesFile, analyzeSalesText } from './services/geminiService';
import { AnalysisResult, AppStatus, UserPersona } from './types';

// --- [권한 체크 통합 컴포넌트 시작] ---
function AuthGate({ children }: { children: React.ReactNode }) {
  const [email, setEmail] = useState<string | null>(localStorage.getItem('userEmail'));
  const [authorizedUsers, setAuthorizedUsers] = useState<Record<string, string>>({});
  const [isLoading, setIsLoading] = useState(true);

  // 1. 구글 시트에서 데이터 로드
  useEffect(() => {
    const SHEET_ID = '130AsLYhQu0ZU6e8LY_28arUHTg5WUODHQg3GlR_3agM';
    const url = `https://docs.google.com/spreadsheets/d/${SHEET_ID}/gviz/tq?tqx=out:csv`;

    const fetchSheetData = async () => {
      try {
        const response = await fetch(url);
        const csvText = await response.text();
        const rows = csvText.split('\n').slice(1);
        const data: Record<string, string> = {};
        
        rows.forEach(row => {
          const columns = row.split(',').map(col => col.replace(/"/g, '').trim());
          const mail = columns[0]?.toLowerCase(); // A열: 이메일
          const date = columns[1];               // B열: 만료일 (YYYY-MM-DD)
          if (mail) data[mail] = date || '2099-12-31';
        });
        
        setAuthorizedUsers(data);
        setIsLoading(false);
      } catch (error) {
        console.error("Auth Load Failed:", error);
        setIsLoading(false);
      }
    };
    fetchSheetData();
  }, []);

  // 2. 이메일 입력 로직
  useEffect(() => {
    if (!email && !isLoading) {
      const input = prompt("긱어스 결제 시 사용한 이메일을 입력해주세요:");
      if (input) {
        const cleaned = input.trim().toLowerCase();
        localStorage.setItem('userEmail', cleaned);
        setEmail(cleaned);
      }
    }
  }, [email, isLoading]);

  if (isLoading) return <div style={{ display: 'flex', height: '100vh', alignItems: 'center', justifyContent: 'center', backgroundColor: '#020617', color: '#6366f1' }}>권한 확인 중...</div>;

  const today = new Date().toISOString().split('T')[0];
  const expiryDate = email ? authorizedUsers[email] : null;
  const isAuthorized = expiryDate && expiryDate >= today;

  if (isAuthorized) return <>{children}</>;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh', textAlign: 'center', backgroundColor: '#020617', color: 'white', fontFamily: 'sans-serif', padding: '20px' }}>
      <div style={{ padding: '40px', backgroundColor: '#0f172a', borderRadius: '24px', border: '1px solid rgba(255,255,255,0.1)', maxWidth: '400px' }}>
        <h1 style={{ fontSize: '24px', fontWeight: '900', marginBottom: '16px' }}>{expiryDate ? "⏳ 이용 기간 만료" : "🔐 승인 대기 중"}</h1>
        <p style={{ color: '#94a3b8', fontSize: '14px', marginBottom: '32px' }}>{expiryDate ? `${email}님의 이용 기간이 ${expiryDate}에 종료되었습니다.` : "입력하신 이메일이 결제 명단에 없거나 승인 전입니다."}</p>
        <button onClick={() => { localStorage.removeItem('userEmail'); window.location.reload(); }} style={{ width: '100%', padding: '14px', cursor: 'pointer', backgroundColor: '#4f46e5', color: 'white', border: 'none', borderRadius: '12px', fontWeight: 'bold' }}>이메일 다시 입력하기</button>
      </div>
    </div>
  );
}
// --- [권한 체크 통합 컴포넌트 끝] ---

type MainTab = 'PRE_MEETING' | 'SALES_ANALYSIS' | 'AI_COACH';

const ANALYSIS_STEPS = [
  "데이터 로드 및 무결성 검사 중...",
  "닐 래컴의 SPIN 프레임워크 엔진 가동...",
  "전체 대화 맥락 및 페르소나 매칭 중...",
  "심리적 트리거 및 설득의 6원칙 탐색...",
  "Growth Point 및 개선 방안 도출...",
  "전문가급 세일즈 스크립트 설계 중...",
  "리포트 시각화 데이터 생성 중...",
  "최종 검토 및 데이터 패키징 중..."
];

function App() {
  const [mainTab, setMainTab] = useState<MainTab>('PRE_MEETING');
  const [status, setStatus] = useState<AppStatus>(AppStatus.IDLE);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string>('');
  const [isServerBusy, setIsServerBusy] = useState(false);
  const [loadingLogs, setLoadingLogs] = useState<{message: string, timestamp: number}[]>([]);
  const [currentStepIdx, setCurrentStepIdx] = useState(0);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [persona, setPersona] = useState<UserPersona>({ name: '', background: '', goal: '', isActive: false });
  const [isPersonaModalOpen, setIsPersonaModalOpen] = useState(false);
  const [lastAction, setLastAction] = useState<(() => Promise<AnalysisResult>) | null>(null);

  useEffect(() => {
    const saved = localStorage.getItem('sales_persona_v2');
    if (saved) { try { setPersona(JSON.parse(saved)); } catch (e) { console.error(e); } }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [mainTab, status]);

  useEffect(() => {
    let interval: any;
    let timer: any;
    if (status === AppStatus.ANALYZING) {
      setElapsedTime(0);
      interval = setInterval(() => { setCurrentStepIdx(prev => (prev + 1) % ANALYSIS_STEPS.length); }, 2000);
      timer = setInterval(() => { setElapsedTime(prev => prev + 1); }, 1000);
    }
    return () => { clearInterval(interval); clearInterval(timer); };
  }, [status]);

  const addLog = useCallback((msg: string) => {
    setLoadingLogs(prev => [...prev, { message: msg, timestamp: Date.now() }]);
  }, []);

  const handleSavePersona = (newPersona: UserPersona) => {
    setPersona(newPersona);
    localStorage.setItem('sales_persona_v2', JSON.stringify(newPersona));
  };

  const processAnalysis = async (method: () => Promise<AnalysisResult>) => {
    setLastAction(() => method);
    setStatus(AppStatus.ANALYZING);
    setLoadingLogs([]);
    setErrorMsg('');
    setIsServerBusy(false);
    try {
      const data = await method();
      setResult(data);
      setStatus(AppStatus.SUCCESS);
    } catch (err: any) {
      const msg = String(err.message || err);
      const isQuota = msg.includes("429") || msg.toLowerCase().includes("quota");
      const is503 = msg.includes("503") || msg.toLowerCase().includes("high demand");
      setIsServerBusy(isQuota || is503);
      setErrorMsg(isQuota ? "API 할당량 초과" : (is503 ? "서버 혼잡" : (err.message || '오류 발생')));
      setStatus(AppStatus.ERROR);
    }
  };

  const handleRetry = () => lastAction ? processAnalysis(lastAction) : setStatus(AppStatus.IDLE);

  return (
    <AuthGate>
      <div className="min-h-screen bg-[#020617] text-slate-200 font-sans">
        <header className="sticky top-0 z-[60] w-full border-b border-white/5 bg-[#020617]/90 backdrop-blur-xl px-4 sm:px-12 h-16 sm:h-20 flex items-center justify-between">
          <div className="flex items-center gap-2 sm:gap-3 cursor-pointer" onClick={() => { setStatus(AppStatus.IDLE); setResult(null); }}>
             <div className="w-8 h-8 sm:w-9 sm:h-9 bg-indigo-600 rounded-lg flex items-center justify-center shadow-lg shadow-indigo-600/20">
                <svg className="w-4 h-4 sm:w-5 sm:h-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><path d="M12 20V4m-6 16v-6m12 6V10" /></svg>
             </div>
             <div className="flex flex-col">
              <h1 className="text-xs sm:text-lg font-black text-white tracking-tighter uppercase leading-none">SALES <span className="text-cyan-400">DIAGNOSTICS</span></h1>
              <span className="text-[7px] sm:text-[10px] font-bold text-slate-500 tracking-[0.2em] mt-0.5 sm:mt-1 uppercase">Professional AI Edition V2.2</span>
             </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 py-8 relative">
          {status === AppStatus.IDLE && (
            <div className="flex flex-col items-center gap-10">
                <div className="flex flex-col lg:flex-row items-center justify-center w-full max-w-4xl gap-6 sm:gap-8 mb-10">
                    <nav className="flex p-1 bg-slate-900/50 rounded-2xl border border-white/5 shadow-xl w-full max-w-lg">
                        {['PRE_MEETING', 'SALES_ANALYSIS', 'AI_COACH'].map((id) => (
                            <button key={id} onClick={() => setMainTab(id as MainTab)} className={`flex-1 py-2.5 sm:py-3 px-2 rounded-xl text-[10px] sm:text-xs font-black ${mainTab === id ? 'bg-indigo-600 text-white' : 'text-slate-500'}`}>{id === 'PRE_MEETING' ? '미팅 전략' : id === 'SALES_ANALYSIS' ? '정밀 진단' : 'AI 코치'}</button>
                        ))}
                    </nav>
                    <button onClick={() => setIsPersonaModalOpen(true)} className="relative w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-indigo-600 flex flex-col items-center justify-center shadow-xl">
                        <span className="text-[7px] text-white font-black">MY PERSONA</span>
                    </button>
                </div>
                <div className="w-full max-w-4xl">
                  {mainTab === 'PRE_MEETING' && <PreMeetingView persona={persona} />}
                  {mainTab === 'SALES_ANALYSIS' && <FileUpload onFileSelect={(f) => processAnalysis(() => analyzeSalesFile(f, persona, addLog))} onScriptSelect={(s) => processAnalysis(() => analyzeSalesText(s, persona, addLog))} disabled={false} />}
                  {mainTab === 'AI_COACH' && <ChatInterface persona={persona} />}
                </div>
            </div>
          )}

          {status === AppStatus.ANALYZING && <div className="py-20 text-center">분석 중... ({elapsedTime}초)</div>}
          {status === AppStatus.SUCCESS && result && <AnalysisView result={result} onReset={() => setStatus(AppStatus.IDLE)} />}
          {status === AppStatus.ERROR && <div className="py-20 text-center"><h2 className="text-white">{errorMsg}</h2><button onClick={handleRetry} className="mt-4 px-6 py-2 bg-indigo-600 rounded-lg">다시 시도</button></div>}
        </main>

        <PersonaSettings isOpen={isPersonaModalOpen} onClose={() => setIsPersonaModalOpen(false)} onSave={handleSavePersona} initialData={persona} />
      </div>
    </AuthGate>
  );
}

export default App;
