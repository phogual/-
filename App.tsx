
import React, { useState, useEffect, useCallback } from 'react';
import { FileUpload } from './components/FileUpload';
import { AnalysisView } from './components/AnalysisView';
import { ChatInterface } from './components/ChatInterface';
import { PreMeetingView } from './components/PreMeetingView';
import { PersonaSettings } from './components/PersonaSettings';
import { analyzeSalesFile, analyzeSalesText } from './services/geminiService';
import { AnalysisResult, AppStatus, UserPersona } from './types';

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
  
  // 마지막 실행한 분석 작업을 기억하여 재시도 가능하게 함
  const [lastAction, setLastAction] = useState<(() => Promise<AnalysisResult>) | null>(null);

  useEffect(() => {
    const saved = localStorage.getItem('sales_persona_v2');
    if (saved) {
        try { setPersona(JSON.parse(saved)); } catch (e) { console.error(e); }
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [mainTab, status]);

  useEffect(() => {
    let interval: any;
    let timer: any;
    if (status === AppStatus.ANALYZING) {
      setElapsedTime(0);
      interval = setInterval(() => {
        setCurrentStepIdx(prev => (prev + 1) % ANALYSIS_STEPS.length);
      }, 2000);
      timer = setInterval(() => {
        setElapsedTime(prev => prev + 1);
      }, 1000);
    } else {
      setCurrentStepIdx(0);
      setElapsedTime(0);
    }
    return () => {
      clearInterval(interval);
      clearInterval(timer);
    };
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
      console.error("Analysis Error:", err);
      const msg = String(err.message || err);
      // 429 및 503 오류를 '서버 혼잡' 상태로 분류
      const isQuotaError = msg.includes("429") || msg.toLowerCase().includes("quota") || msg.toLowerCase().includes("exhausted");
      const is503 = msg.includes("503") || msg.toLowerCase().includes("high demand") || msg.toLowerCase().includes("unavailable");
      
      setIsServerBusy(isQuotaError || is503);
      setErrorMsg(isQuotaError 
        ? "API 이용 할당량을 초과했습니다. 무료 티어 사용자의 경우 잠시 후(약 1분 뒤) 다시 시도해 주세요." 
        : (is503 ? "현재 AI 서버 이용자가 몰려 지연되고 있습니다. 잠시 후 다시 시도 버튼을 눌러주세요." : (err.message || '분석 중 예상치 못한 오류가 발생했습니다.')));
      setStatus(AppStatus.ERROR);
    }
  };

  const handleRetry = () => {
    if (lastAction) {
      processAnalysis(lastAction);
    } else {
      setStatus(AppStatus.IDLE);
    }
  };

  return (
    <div className="min-h-screen bg-[#020617] text-slate-200 font-sans">
      <header className="sticky top-0 z-[60] w-full border-b border-white/5 bg-[#020617]/90 backdrop-blur-xl px-4 sm:px-12 h-16 sm:h-20 flex items-center justify-between">
        <div className="flex items-center gap-2 sm:gap-3 cursor-pointer" onClick={() => { setStatus(AppStatus.IDLE); setResult(null); }}>
           <div className="w-8 h-8 sm:w-9 sm:h-9 bg-indigo-600 rounded-lg flex items-center justify-center shadow-lg shadow-indigo-600/20">
              <svg className="w-4 h-4 sm:w-5 sm:h-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><path d="M12 20V4m-6 16v-6m12 6V10" /></svg>
           </div>
           <div className="flex flex-col">
            <h1 className="text-xs sm:text-lg font-black text-white tracking-tighter uppercase leading-none">
                SALES <span className="text-cyan-400">DIAGNOSTICS</span>
            </h1>
            <span className="text-[7px] sm:text-[10px] font-bold text-slate-500 tracking-[0.2em] mt-0.5 sm:mt-1 uppercase">Professional AI Edition V2.2</span>
           </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8 relative">
        {status === AppStatus.IDLE && (
          <div className="flex flex-col items-center gap-10">
            <div className="w-full flex flex-col items-center relative">
                <div className="flex flex-col lg:flex-row items-center justify-center w-full max-w-4xl gap-6 sm:gap-8 mb-10">
                    <nav className="flex p-1 bg-slate-900/50 rounded-2xl border border-white/5 shadow-xl w-full max-w-lg overflow-x-auto no-scrollbar">
                        {[
                            { id: 'PRE_MEETING', label: '미팅 전략' },
                            { id: 'SALES_ANALYSIS', label: '정밀 진단' },
                            { id: 'AI_COACH', label: 'AI 코치' }
                        ].map((tab) => (
                            <button
                            key={tab.id}
                            onClick={() => setMainTab(tab.id as MainTab)}
                            className={`flex-1 py-2.5 sm:py-3 px-2 rounded-xl text-[10px] sm:text-xs font-black transition-all whitespace-nowrap ${mainTab === tab.id ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-500 hover:bg-white/5'}`}
                            >
                            {tab.label}
                            </button>
                        ))}
                    </nav>

                    <div className="lg:absolute lg:right-0 flex flex-col items-center">
                        <button 
                            onClick={() => setIsPersonaModalOpen(true)}
                            className="group relative flex flex-col items-center transition-all duration-500 hover:scale-110 active:scale-95"
                        >
                            <div className="absolute inset-0 bg-purple-600/30 blur-[25px] rounded-full scale-110 opacity-60 group-hover:opacity-100 transition-opacity"></div>
                            <div className="relative w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-white p-[1.5px] shadow-[0_0_20px_rgba(168,85,247,0.3)] overflow-hidden">
                                <div className="w-full h-full rounded-full bg-gradient-to-b from-[#6366f1] via-[#4c1d95] to-[#1e1b4b] flex flex-col items-center justify-center relative">
                                    <div className="mb-0">
                                        <svg className="w-6 h-6 sm:w-7 sm:h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                                        </svg>
                                    </div>
                                    <div className="flex flex-col items-center">
                                        <span className="text-[7px] sm:text-[8.5px] font-black text-white leading-none tracking-tight uppercase">MY PERSONA</span>
                                        <span className="text-[5px] sm:text-[6.5px] font-bold text-cyan-400 mt-0.5 uppercase">AI CORE</span>
                                    </div>
                                    <div className="absolute inset-0 bg-gradient-to-tr from-white/10 to-transparent opacity-40"></div>
                                </div>
                            </div>
                            <span className="mt-2 text-[8px] sm:text-[9px] font-black text-cyan-400 uppercase tracking-[0.4em] drop-shadow-[0_0_6px_rgba(34,211,238,0.5)]">
                                {persona.name || 'PREMIUM EXPERT'}
                            </span>
                        </button>
                    </div>
                </div>

                <div className="w-full max-w-4xl">
                  {mainTab === 'PRE_MEETING' && <PreMeetingView persona={persona} />}
                  {mainTab === 'SALES_ANALYSIS' && <FileUpload onFileSelect={(f) => processAnalysis(() => analyzeSalesFile(f, persona, addLog))} onScriptSelect={(s) => processAnalysis(() => analyzeSalesText(s, persona, addLog))} disabled={false} />}
                  {mainTab === 'AI_COACH' && <ChatInterface persona={persona} />}
                </div>
            </div>
          </div>
        )}

        {status === AppStatus.ANALYZING && (
          <div className="py-20 flex flex-col items-center text-center max-w-2xl mx-auto px-4">
            <div className="relative mb-12">
                <div className="w-24 h-24 border-4 border-indigo-500/10 border-t-indigo-500 rounded-full animate-spin shadow-[0_0_40px_rgba(99,102,241,0.2)]"></div>
                <div className="absolute inset-0 flex items-center justify-center">
                    <svg className="w-8 h-8 text-indigo-400 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                </div>
            </div>
            
            <div className="space-y-4 mb-10">
                <h2 className="text-2xl sm:text-3xl font-black text-white uppercase tracking-tighter">심층 심리 분석 엔진 가동 중</h2>
                <div className="flex flex-col items-center">
                    <p className="text-indigo-400 font-bold text-base sm:text-lg h-10 transition-all duration-500 animate-fade-in" key={currentStepIdx}>
                        {ANALYSIS_STEPS[currentStepIdx]}
                    </p>
                    <div className="w-full max-w-[240px] sm:max-w-xs h-1.5 bg-slate-900 rounded-full mt-4 overflow-hidden border border-white/5">
                        <div 
                            className="h-full bg-indigo-500 transition-all duration-700 ease-out shadow-[0_0_10px_#6366f1]" 
                            style={{ width: `${Math.min(98, ((currentStepIdx + 1) / ANALYSIS_STEPS.length) * 100 + (elapsedTime / 1))}%` }}
                        ></div>
                    </div>
                </div>
            </div>

            {elapsedTime > 30 && (
              <div className="mb-8 p-4 bg-amber-500/10 border border-amber-500/30 rounded-2xl animate-fade-in">
                <p className="text-amber-400 text-xs font-bold leading-relaxed">
                  ⚠️ 대용량 대화를 고속 분석 중입니다. <br/>
                  완벽한 리포트를 위해 잠시만 더 기다려 주세요. ({elapsedTime}초 경과)
                </p>
              </div>
            )}

            <div className="w-full bg-slate-900/40 border border-white/5 rounded-3xl p-6 text-left shadow-inner">
                <div className="flex items-center justify-between mb-4 border-b border-white/5 pb-2">
                    <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Deep Scan Log</span>
                    <span className="text-[10px] font-bold text-indigo-500/60 uppercase">Step {currentStepIdx + 1}/8</span>
                </div>
                <div className="h-32 overflow-y-auto custom-scrollbar space-y-2">
                    {loadingLogs.length === 0 ? (
                        <p className="text-[11px] font-mono text-slate-600 italic">대화 전문을 독해하고 심리 패턴을 추출하는 중입니다...</p>
                    ) : (
                        loadingLogs.map((log, i) => (
                            <p key={i} className="text-[11px] font-mono text-indigo-300/80 animate-fade-in flex gap-3">
                                <span className="opacity-30">[{new Date(log.timestamp).toLocaleTimeString()}]</span>
                                <span>{log.message}</span>
                            </p>
                        ))
                    )}
                </div>
            </div>
          </div>
        )}

        {status === AppStatus.SUCCESS && result && (
          <AnalysisView result={result} onReset={() => setStatus(AppStatus.IDLE)} />
        )}

        {status === AppStatus.ERROR && (
          <div className="py-20 text-center animate-fade-in max-w-md mx-auto">
             <div className="w-20 h-20 bg-amber-500/10 rounded-full flex items-center justify-center mx-auto mb-8 border border-amber-500/20">
                <svg className="w-10 h-10 text-amber-500 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
             </div>
             <h2 className="text-2xl font-black text-white mb-4 uppercase tracking-tight">분석에 실패했습니다</h2>
             <div className="bg-slate-900/60 p-6 rounded-2xl border border-white/5 mb-10">
                <p className="text-slate-300 text-sm leading-relaxed mb-4">
                    {isServerBusy ? "현재 AI 서버가 매우 혼잡하거나 할당량이 초과되었습니다. 안정적인 분석을 위해 다시 시도해 주세요." : errorMsg}
                </p>
                {isServerBusy && (
                    <span className="inline-block px-3 py-1 bg-amber-500/20 text-amber-500 text-[10px] font-black rounded-full uppercase tracking-widest border border-amber-500/30">Quota/Server Busy (429/503)</span>
                )}
             </div>
             <div className="flex flex-col gap-3">
                <button onClick={handleRetry} className="w-full py-4 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl font-black shadow-xl shadow-indigo-600/20 active:scale-95 transition-all">다시 시도</button>
                <button onClick={() => setStatus(AppStatus.IDLE)} className="w-full py-4 bg-slate-800 text-slate-400 rounded-xl font-bold border border-white/5 hover:text-white transition-all">메인으로 돌아가기</button>
             </div>
          </div>
        )}
      </main>

      <PersonaSettings 
        isOpen={isPersonaModalOpen} 
        onClose={() => setIsPersonaModalOpen(false)} 
        onSave={handleSavePersona} 
        initialData={persona}
      />
    </div>
  );
}

export default App;
