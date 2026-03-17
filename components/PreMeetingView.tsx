
import React, { useState, useRef, useEffect } from 'react';
import { PreMeetingStrategy, UserPersona, PersuasionTactic, CharlieMorganInsight, CialdiniInsight } from '../types';
import { generatePreMeetingStrategy } from '../services/geminiService';

interface PreMeetingViewProps {
  persona: UserPersona;
}

const getDynamicFontSize = (text: string = '', baseSize: number, minSize: number, maxLength: number) => {
  const len = text.length;
  if (len === 0) return `${baseSize}px`;
  if (len < maxLength * 0.4) return `${baseSize * 1.15}px`;
  if (len <= maxLength) return `${baseSize}px`;
  const ratio = Math.max(minSize / baseSize, 1 - (len - maxLength) / (maxLength * 1.3));
  return `${baseSize * ratio}px`;
};

const CharlieMorganInsightPage: React.FC<{ insight: CharlieMorganInsight }> = ({ insight }) => (
    <div className="flex flex-col gap-4 flex-1">
        <div className="flex justify-between items-start">
            <h1 className="text-[52px] font-black text-amber-500 leading-[0.85] tracking-tighter italic uppercase">Morgan's Insight</h1>
            <div className="text-right pt-2">
                <span className="text-[10px] text-slate-500 font-black uppercase tracking-[0.2em] mb-1 block">시스템 청사진</span>
                <div className="bg-amber-500/10 px-6 py-2 rounded-full border-2 border-amber-500/20 text-[11px] font-black text-amber-500 uppercase tracking-widest">사전 미팅 전략</div>
            </div>
        </div>
        <div className="grid grid-cols-2 gap-6 flex-1 overflow-hidden">
            <div className="flex flex-col gap-5">
                <div className="bg-slate-900/40 p-8 rounded-[35px] border border-white/5 flex flex-col gap-3 shadow-lg h-1/2 overflow-hidden justify-center">
                    <h3 className="text-[12px] font-black text-amber-500 uppercase tracking-[0.4em]">심층 고통 분석 (Pain)</h3>
                    <p className="leading-[1.6] text-slate-200 font-medium" style={{ fontSize: getDynamicFontSize(insight.deepPain, 17, 13, 180) }}>{insight.deepPain}</p>
                </div>
                <div className="bg-slate-900/40 p-8 rounded-[35px] border border-white/5 flex flex-col gap-3 shadow-lg h-1/2 overflow-hidden justify-center">
                    <h3 className="text-[12px] font-black text-amber-500 uppercase tracking-[0.4em]">현재와 미래의 격차 (Gap)</h3>
                    <p className="leading-[1.6] text-slate-200 font-medium" style={{ fontSize: getDynamicFontSize(insight.gapDefinition, 17, 13, 180) }}>{insight.gapDefinition}</p>
                </div>
            </div>
            <div className="bg-[#0b1018] p-10 rounded-[45px] border-2 border-amber-500/20 shadow-2xl flex flex-col gap-6 overflow-hidden justify-between">
                <div className="flex flex-col gap-3">
                    <h2 className="text-2xl font-black text-white italic uppercase tracking-tighter">가교 전략 (The Bridge)</h2>
                    <p className="leading-[1.4] text-white font-bold italic" style={{ fontSize: getDynamicFontSize(insight.bridgePositioning, 24, 18, 100) }}>“{insight.bridgePositioning}”</p>
                </div>
                <div className="flex flex-col gap-3 border-t-2 border-white/10 pt-6">
                    <h3 className="text-[12px] font-black text-indigo-400 uppercase tracking-[0.4em]">반론 돌파 및 마인드셋 전략</h3>
                    <p className="leading-[1.5] text-slate-300 font-medium" style={{ fontSize: getDynamicFontSize(insight.objectionStrategy, 17, 13, 140) }}>{insight.objectionStrategy}</p>
                </div>
                <div className="mt-auto bg-amber-600 p-6 rounded-[25px] shadow-2xl"><p className="text-white font-black text-center text-[18px] tracking-tight uppercase italic leading-tight">“당신의 의지는 죄가 없습니다. 시스템을 교체하십시오.”</p></div>
            </div>
        </div>
    </div>
);

const CialdiniInsightPage: React.FC<{ insight: CialdiniInsight }> = ({ insight }) => (
    <div className="flex flex-col flex-1 overflow-hidden h-full">
        <div className="flex flex-col mb-1 shrink-0 pt-2">
            <h1 className="text-[44px] font-black text-cyan-400 leading-none tracking-tighter italic uppercase">초전설득 5원칙</h1>
            <p className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] mt-1 italic">Pre-suasion 5 Logic Questions</p>
        </div>
        <div className="grid grid-cols-12 gap-4 flex-1 overflow-hidden pt-2">
            <div className="col-span-4 flex flex-col gap-3 h-full overflow-hidden">
                <div className="bg-slate-900/40 p-6 rounded-[25px] border border-white/5 flex flex-col gap-3 shadow-lg flex-1 overflow-hidden justify-center">
                    <h3 className="text-[10px] font-black text-cyan-400 uppercase tracking-[0.4em] mb-0.5">아키텍처 (Architecture)</h3>
                    <p className="leading-[1.4] text-slate-200 font-medium italic" style={{ fontSize: getDynamicFontSize(insight.preSuasionStrategy, 15, 11, 130) }}>{insight.preSuasionStrategy}</p>
                </div>
                <div className="bg-slate-900/40 p-6 rounded-[25px] border border-white/5 flex flex-col gap-3 shadow-lg flex-1 overflow-hidden justify-center">
                    <h3 className="text-[10px] font-black text-cyan-400 uppercase tracking-[0.4em] mb-0.5">프레이밍 논리 (Framing)</h3>
                    <p className="leading-[1.4] text-slate-200 font-medium italic" style={{ fontSize: getDynamicFontSize(insight.framingLogic, 15, 11, 130) }}>{insight.framingLogic}</p>
                </div>
                <div className="bg-cyan-950/20 p-4 rounded-[20px] border border-cyan-400/10 text-center shrink-0"><p className="text-[10px] text-cyan-300 font-black italic tracking-tight uppercase">“심리학은 도구입니다. 프레임을 먼저 설치하십시오.”</p></div>
            </div>
            <div className="col-span-8 flex flex-col gap-1 overflow-hidden h-full">
                <h3 className="text-[9px] font-black text-slate-500 uppercase tracking-[0.4em] mb-1 pl-2 shrink-0 italic">심리적 점화 5대 원칙 기반 질문</h3>
                <div className="flex flex-col justify-between flex-1 pb-4 gap-2.5 overflow-hidden">
                    {(insight.structuredQuestions || []).slice(0, 5).map((q, i) => (
                        <div key={i} className="bg-cyan-400/5 p-4.5 rounded-[20px] border-2 border-cyan-400/20 relative overflow-hidden group hover:bg-cyan-400/10 transition-all shadow-md flex-1 flex items-center min-h-0">
                            <div className="absolute top-0 left-0 w-1.5 h-full bg-cyan-400"></div>
                            <div className="flex items-center gap-4 w-full">
                                <div className="w-9 h-9 rounded-xl bg-cyan-400/10 border border-cyan-400/20 flex items-center justify-center shrink-0"><span className="text-cyan-400 font-black text-base italic">{i+1}</span></div>
                                <div className="flex flex-col min-w-0">
                                    <span className="text-[9px] font-black text-cyan-500/70 uppercase tracking-widest leading-none mb-1">{q.principle}</span>
                                    <p className="text-white font-bold italic leading-tight" style={{ fontSize: getDynamicFontSize(q.question, 17, 13, 85) }}>“{q.question}”</p>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    </div>
);

const SpinQuestionSection: React.FC<{ title: string; questions: string[]; analysis: string; score: number; color: string; bgColor: string; label: string }> = ({ title, questions, analysis, score, color, bgColor, label }) => (
    <div className={`flex-1 p-8 rounded-[40px] border border-white/5 flex flex-col gap-5 bg-[#0b1018] relative overflow-hidden group shadow-inner transition-transform hover:scale-[1.01]`}>
        <div className="flex items-center justify-between">
            <h4 className={`text-[16px] font-black uppercase tracking-tighter ${color}`}>{title}</h4>
            <div className="flex items-center gap-3">
                <span className={`px-4 py-1.5 rounded-full text-[10px] font-black text-white ${label}`}>{title.split(' ')[0]}</span>
            </div>
        </div>
        <div className="grid grid-cols-1 gap-4 flex-1">
            <div className="space-y-3">
                {(questions || []).slice(0, 4).map((q, i) => (
                    <div key={i} className="flex gap-4 items-start group/item">
                        <span className={`text-[18px] font-black opacity-30 shrink-0 mt-0.5 ${color}`}>{i + 1}</span>
                        <p className="text-slate-200 font-medium leading-relaxed italic group-hover/item:text-white transition-colors" style={{ fontSize: getDynamicFontSize(q, 16, 13, 90) }}>“{q}”</p>
                    </div>
                ))}
            </div>
            <div className="bg-slate-900/60 p-5 rounded-2xl border border-white/5 flex flex-col gap-2 mt-auto">
                <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">핵심 개선점</span>
                <p className="text-slate-300 text-sm leading-relaxed font-medium italic" style={{ fontSize: getDynamicFontSize(analysis, 14, 11, 150) }}>{analysis || "분석 대기 중"}</p>
            </div>
        </div>
    </div>
);

const PersuasionTacticCard: React.FC<{ tactic: PersuasionTactic }> = ({ tactic }) => (
    <div className="grid grid-cols-12 gap-4 bg-slate-900/30 p-6 rounded-[30px] border border-white/5 relative overflow-hidden group hover:bg-slate-900/50 transition-all shadow-lg items-center">
        <div className="absolute top-0 left-0 w-1 h-full bg-emerald-500/40"></div>
        <div className="col-span-3 flex flex-col justify-center"><span className="text-[12px] font-black text-emerald-400 uppercase tracking-tighter mb-1">{tactic.principle}</span></div>
        <div className="col-span-4 flex flex-col gap-1 justify-center overflow-hidden"><span className="text-[9px] font-black text-slate-500 uppercase tracking-widest border-l-2 border-slate-700 pl-3 leading-none mb-1">CONCEPT</span><p className="text-slate-300 text-xs leading-relaxed pl-3 line-clamp-2">{tactic.description}</p></div>
        <div className="col-span-5 flex flex-col gap-1 justify-center overflow-hidden"><span className="text-[9px] font-black text-emerald-500 uppercase tracking-widest border-l-2 border-emerald-800 pl-3 leading-none mb-1">SCRIPT</span><p className="text-white font-bold leading-tight pl-3 italic text-sm">“{tactic.script}”</p></div>
    </div>
);

export const PreMeetingView: React.FC<PreMeetingViewProps> = ({ persona }) => {
    const [result, setResult] = useState<PreMeetingStrategy | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [loadingMessage, setLoadingMessage] = useState('Designing Strategy...');
    const [context, setContext] = useState('');
    const [file, setFile] = useState<File | null>(null);
    const [dragActive, setDragActive] = useState(false);
    const [isPdfProcessing, setIsPdfProcessing] = useState(false);
    const [pdfProgress, setPdfProgress] = useState(0);
    const [scale, setScale] = useState(1);
    const [editableName, setEditableName] = useState('작성자 성함');
    const fileInputRef = useRef<HTMLInputElement>(null);
    const pageRefs = [
        useRef<HTMLDivElement>(null), 
        useRef<HTMLDivElement>(null), 
        useRef<HTMLDivElement>(null), 
        useRef<HTMLDivElement>(null)
    ];

    useEffect(() => {
        const updateScale = () => { const containerWidth = window.innerWidth - 32; setScale(containerWidth < 1131 ? containerWidth / 1131 : 1); };
        window.addEventListener('resize', updateScale);
        updateScale();
        return () => window.removeEventListener('resize', updateScale);
    }, []);

    useEffect(() => { if (result?.contactInfo?.name) setEditableName(result.contactInfo.name); }, [result]);

    const handleDrag = (e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); if (e.type === "dragenter" || e.type === "dragover") setDragActive(true); else if (e.type === "dragleave") setDragActive(false); };
    const handleDrop = (e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); setDragActive(false); if (e.dataTransfer.files && e.dataTransfer.files[0]) { setFile(e.dataTransfer.files[0]); setContext(''); } };

    const [errorMsg, setErrorMsg] = useState<string | null>(null);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!context.trim() && !file) return;
        setIsLoading(true);
        setErrorMsg(null);
        setLoadingMessage('전략 설계 엔진 가동 중...');
        try {
            const strategy = await generatePreMeetingStrategy(file || context, persona, (msg) => setLoadingMessage(msg));
            setResult(strategy);
        } catch (err: any) { 
            console.error("Strategy Generation Error:", err);
            setErrorMsg(err.message || "전략 수립 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.");
        }
        finally { setIsLoading(false); }
    };

    const handleProcessPDF = async () => {
        const html2canvas = (window as any).html2canvas;
        const jspdfObj = (window as any).jspdf;
        if (!html2canvas || !jspdfObj || isPdfProcessing) {
            if (!html2canvas || !jspdfObj) alert("라이브러리가 아직 로드되지 않았습니다.");
            return;
        }
        setIsPdfProcessing(true);
        setPdfProgress(0);
        try {
            const pdf = new jspdfObj.jsPDF({
                orientation: 'l',
                unit: 'mm',
                format: 'a4',
                compress: true
            });
            const originalScrollPos = window.scrollY;
            window.scrollTo(0, 0);
            
            for (let i = 0; i < 4; i++) {
                setPdfProgress(i + 1);
                
                // 렌더링 및 스타일 적용 대기
                await new Promise(r => setTimeout(r, 800));
                
                const targetId = `pre-page-${i}`;
                const element = document.getElementById(targetId);
                if (!element) {
                    console.warn(`Page element not found: ${targetId}`);
                    continue;
                }
                
                const canvas = await html2canvas(element, { 
                    scale: 2, 
                    backgroundColor: '#0b1018', 
                    useCORS: true, 
                    logging: false,
                    allowTaint: false,
                    onclone: (clonedDoc) => {
                        const clonedPage = clonedDoc.getElementById(targetId);
                        if (clonedPage) {
                            clonedPage.style.transform = 'none';
                            clonedPage.style.position = 'relative';
                            clonedPage.style.display = 'flex';
                            clonedPage.style.visibility = 'visible';
                            clonedPage.style.opacity = '1';
                        }
                    }
                });
                
                if (i > 0) pdf.addPage();
                
                const imgData = canvas.toDataURL('image/jpeg', 0.8);
                pdf.addImage(imgData, 'JPEG', 0, 0, 297, 210, undefined, 'FAST');
                
                // 메모리 해제
                canvas.width = 0;
                canvas.height = 0;
            }
            pdf.save(`Sales_Strategy_${editableName}.pdf`);
            window.scrollTo(0, originalScrollPos);
        } catch (e) { 
            console.error("PDF Error:", e);
            alert("PDF 생성 실패. 브라우저 메모리가 부족할 수 있습니다."); 
        } finally { 
            setIsPdfProcessing(false);
            setPdfProgress(0);
        }
    };

    const PageWrapper = ({ children, index }: React.PropsWithChildren<{ index: number }>) => (
        <div className="flex justify-center w-full mb-10" style={{ height: `${800 * scale}px` }}>
            <div 
                id={`pre-page-${index}`}
                ref={pageRefs[index]} 
                className="bg-[#0b1018] text-white overflow-hidden shadow-2xl relative flex flex-col origin-top shrink-0" 
                style={{ width: '1131px', height: '800px', transform: `scale(${scale})` }}
            >
                <div className="flex-1 flex flex-col p-[45px] relative z-10">{children}</div>
                <div className="h-[40px] flex items-center justify-between px-[45px] bg-[#0b1018] border-t border-white/5">
                <div className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse"></div><span className="text-[10px] font-bold tracking-widest text-slate-500 uppercase">세일즈 인텔리전스 엔진 V2.0</span></div>
                <span className="text-[10px] font-bold tracking-widest text-slate-500 uppercase">페이지 {index + 1} / 4</span>
            </div>
            </div>
        </div>
    );

    return (
        <div className="w-full animate-fade-in pb-20">
            {!result ? (
                <div className="bg-slate-900/50 backdrop-blur-md border border-white/10 rounded-3xl p-10 shadow-2xl relative">
                    {isLoading && <div className="absolute inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex flex-col items-center justify-center rounded-3xl"><div className="w-12 h-12 border-4 border-indigo-500/10 border-t-indigo-500 rounded-full animate-spin mb-4"></div><p className="text-white font-black animate-pulse uppercase tracking-widest italic">{loadingMessage}</p></div>}
                    <h2 className="text-3xl font-black text-white mb-2 text-center uppercase tracking-tight italic">STRATEGY <span className="text-indigo-500">REPORT</span></h2>
                    <p className="text-center text-slate-500 text-xs mb-10 uppercase tracking-[0.2em] font-bold font-mono">Will-free system roadmap</p>
                    <form onSubmit={handleSubmit} className="space-y-6">
                        {errorMsg && (
                            <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-2xl animate-fade-in">
                                <p className="text-red-400 text-xs font-bold leading-relaxed text-center">
                                    {errorMsg}
                                </p>
                            </div>
                        )}
                        <div className={`relative group rounded-2xl transition-all duration-300 ${dragActive ? 'ring-2 ring-indigo-500 bg-indigo-500/10' : 'bg-slate-950'}`} onDragEnter={handleDrag} onDragOver={handleDrag} onDragLeave={handleDrag} onDrop={handleDrop}>
                            <textarea value={context} onChange={(e) => setContext(e.target.value)} placeholder="현재 고민, 고객 상황 등을 입력하거나 파일을 드랍하세요..." className={`w-full min-h-[300px] bg-transparent text-white rounded-2xl border border-white/5 p-8 focus:ring-2 focus:ring-indigo-500 transition-all resize-none leading-relaxed text-base ${file ? 'opacity-20' : ''}`} disabled={isLoading || !!file} />
                            {file && <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/90 rounded-2xl border-2 border-indigo-500/50 animate-fade-in p-6"><svg className="w-12 h-12 text-indigo-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg><p className="text-white font-black truncate max-w-full mb-4">{file.name}</p><button type="button" onClick={() => setFile(null)} className="px-6 py-2 bg-slate-800 text-slate-300 rounded-xl font-bold border border-white/10 hover:bg-slate-700">파일 제거</button></div>}
                        </div>
                        <div className="flex gap-4"><button type="button" onClick={() => fileInputRef.current?.click()} className="flex-1 py-4 bg-slate-800 hover:bg-slate-700 text-white rounded-2xl font-bold border border-white/10 transition-all flex items-center justify-center gap-2">로드</button><button type="submit" disabled={isLoading || (!context.trim() && !file)} className="flex-[3] py-4 bg-indigo-600 hover:bg-indigo-500 text-white rounded-2xl font-black shadow-2xl uppercase tracking-widest active:scale-95 disabled:opacity-50">전략 수립 시작</button></div>
                        <input type="file" ref={fileInputRef} onChange={(e) => e.target.files && (setFile(e.target.files[0]), setContext(''))} className="hidden" />
                    </form>
                </div>
            ) : (
                <div className="w-full max-w-[1150px] mx-auto pb-32 px-4">
                    <div className="flex justify-between items-center bg-slate-900/90 p-5 rounded-3xl border border-white/10 sticky top-4 z-[500] backdrop-blur-xl mb-16 shadow-2xl gap-4">
                        <div className="flex flex-col gap-1 w-full max-w-xs"><span className="text-[10px] font-black text-indigo-400 uppercase tracking-widest">Diagnostic Target</span><input type="text" value={editableName} onChange={(e) => setEditableName(e.target.value)} className="bg-slate-950/50 border border-white/10 rounded-lg px-3 py-1.5 text-sm font-bold text-white outline-none w-full" /></div>
                        <div className="flex gap-3 shrink-0"><button onClick={handleProcessPDF} disabled={isPdfProcessing} className="px-8 py-3 bg-indigo-600 hover:bg-indigo-500 rounded-xl text-xs font-black text-white uppercase min-w-[140px]">{isPdfProcessing ? `Page ${pdfProgress}/4...` : 'Download PDF'}</button><button onClick={() => setResult(null)} className="px-8 py-3 bg-slate-800 hover:bg-slate-700 rounded-xl text-xs font-bold text-slate-300 border border-white/10">Restart</button></div>
        </div>
        <PageWrapper index={0}>
            <div className="flex justify-between items-start mb-6"><h1 className="text-[52px] font-black text-white leading-[0.85] tracking-tighter italic uppercase">전략 리포트</h1><div className="text-right pt-1 pr-1"><span className="text-[10px] text-slate-500 font-black uppercase tracking-[0.2em] mb-2 block">프로필</span><div className="bg-[#0f172a]/90 px-8 py-4 rounded-full border-2 border-white/10 text-lg font-black text-white min-w-[240px] flex items-center justify-center min-h-[60px] shadow-2xl">{editableName}</div></div></div>
            <div className="flex flex-col gap-5 flex-1 overflow-hidden">
                <div className="bg-[#0b1018] p-8 rounded-[35px] border border-white/5 flex flex-col gap-3 shadow-inner flex-1 overflow-hidden justify-center"><h3 className="text-[12px] font-black text-indigo-400 uppercase tracking-[0.4em]">고객 맥락 및 비즈니스 환경 분석</h3><div className="text-slate-200 font-medium leading-relaxed" style={{ fontSize: getDynamicFontSize(result.clientContext, 17, 13, 350) }}>{result.clientContext}</div></div>
                <div className="bg-[#0b1018] p-8 rounded-[35px] border border-indigo-500/20 shadow-2xl flex flex-col gap-3 flex-1 overflow-hidden justify-center"><h2 className="text-xl font-black text-white tracking-tight italic mb-1 uppercase">전략적 핵심 가이드라인</h2><div className="text-slate-200 font-medium italic overflow-hidden leading-relaxed" style={{ fontSize: getDynamicFontSize(result.strategySummary, 17, 13, 350) }}>{result.strategySummary}</div></div>
            </div>
        </PageWrapper>

        <PageWrapper index={1}>
            <div className="flex flex-col gap-6 h-full">
                <h2 className="text-[36px] font-black text-white italic tracking-tighter uppercase">SPIN 질문</h2>
                <div className="grid grid-cols-2 gap-6 flex-1 overflow-hidden">
                    <SpinQuestionSection title="SITUATION (상황)" questions={result.spinQuestions.situation} analysis={result.spinAnalysis.situation} score={result.spinScores.situation} color="text-cyan-400" bgColor="bg-cyan-500/5" label="bg-slate-700" />
                    <SpinQuestionSection title="PROBLEM (문제)" questions={result.spinQuestions.problem} analysis={result.spinAnalysis.problem} score={result.spinScores.problem} color="text-blue-400" bgColor="bg-blue-500/5" label="bg-cyan-600" />
                </div>
            </div>
        </PageWrapper>
        <PageWrapper index={2}>
            <div className="flex flex-col gap-6 h-full">
                <h2 className="text-[36px] font-black text-white italic tracking-tighter uppercase">SPIN 질문 (Cont.)</h2>
                <div className="grid grid-cols-2 gap-6 flex-1 overflow-hidden">
                    <SpinQuestionSection title="IMPLICATION (시사)" questions={result.spinQuestions.implication} analysis={result.spinAnalysis.implication} score={result.spinScores.implication} color="text-purple-400" bgColor="bg-purple-500/5" label="bg-violet-700" />
                    <SpinQuestionSection title="NEED-PAYOFF (해결)" questions={result.spinQuestions.needPayoff} analysis={result.spinAnalysis.needPayoff} score={result.spinScores.needPayoff} color="text-pink-400" bgColor="bg-pink-500/5" label="bg-pink-600" />
                </div>
            </div>
        </PageWrapper>
        <PageWrapper index={3}>
            <div className="flex flex-col h-full bg-[#0b1018] rounded-[40px] border-2 border-emerald-500/20 p-12 shadow-2xl relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-1.5 bg-gradient-to-r from-emerald-500/50 via-cyan-500/50 to-emerald-500/50"></div>
                <div className="flex flex-col gap-8">
                    <div className="flex items-center gap-4">
                        <div className="w-2 h-8 bg-emerald-500 rounded-full"></div>
                        <h2 className="text-[28px] font-black text-emerald-400 uppercase tracking-tight italic">핵심 미팅 전략 팁</h2>
                    </div>
                    
                    <div className="flex flex-col gap-8 mt-4">
                        {(result.tips || []).map((tip, i) => (
                            <div key={i} className="flex gap-6 items-start group">
                                <div className="w-2 h-2 rounded-full bg-emerald-500 mt-2.5 shrink-0 shadow-[0_0_10px_rgba(16,185,129,0.5)] group-hover:scale-125 transition-transform"></div>
                                <p className="text-slate-200 font-medium leading-relaxed italic text-[20px] group-hover:text-white transition-colors">
                                    {tip}
                                </p>
                            </div>
                        ))}
                        {(!result.tips || result.tips.length === 0) && (
                            <div className="flex flex-col items-center justify-center py-20 opacity-30">
                                <p className="text-xl italic">전략 팁을 생성 중입니다...</p>
                            </div>
                        )}
                    </div>
                </div>
                
                <div className="mt-auto pt-10 border-t border-white/5 flex justify-center">
                    <p className="text-[12px] font-black text-emerald-500/40 uppercase tracking-[0.5em] italic">Strategic Conclusion & Action Plan</p>
                </div>
            </div>
        </PageWrapper>
                </div>
            )}
        </div>
    );
};
