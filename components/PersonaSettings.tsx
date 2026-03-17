
import React, { useState, useEffect } from 'react';
import { UserPersona } from '../types';

interface PersonaSettingsProps {
  onSave: (persona: UserPersona) => void;
  isOpen: boolean;
  onClose: () => void;
  initialData?: UserPersona;
}

export const PersonaSettings: React.FC<PersonaSettingsProps> = ({ onSave, isOpen, onClose, initialData }) => {
  const [name, setName] = useState(initialData?.name || '');
  const [background, setBackground] = useState(initialData?.background || '');
  const [goal, setGoal] = useState(initialData?.goal || '');
  const [isActive, setIsActive] = useState(initialData?.isActive ?? true);

  useEffect(() => {
    if (initialData) {
        setName(initialData.name || '');
        setBackground(initialData.background || '');
        setGoal(initialData.goal || '');
        setIsActive(initialData.isActive);
    }
  }, [initialData]);

  const handleSave = () => {
    onSave({ name, background, goal, isActive });
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-fade-in">
      <div className="w-full max-w-lg bg-slate-900 border border-purple-500/30 rounded-3xl p-6 sm:p-8 shadow-2xl relative overflow-y-auto max-h-[90vh] no-scrollbar">
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-purple-500 to-transparent"></div>
        
        <div className="flex justify-between items-center mb-6">
            <h3 className="text-xl sm:text-2xl font-black text-white tracking-tight flex items-center gap-3">
                <div className="p-2 bg-purple-500/10 rounded-lg border border-purple-500/20">
                    <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path></svg>
                </div>
                나의 페르소나 설정
            </h3>
            <button onClick={onClose} className="text-slate-500 hover:text-white transition-colors">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
            </button>
        </div>

        <div className="space-y-6">
            <div>
                <label className="block text-[10px] text-purple-400 font-black uppercase tracking-widest mb-2">당신의 이름 (분석 리포트 반영)</label>
                <input 
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="리포트에 표시될 본인의 이름을 입력하세요"
                    className="w-full bg-slate-950 text-white placeholder-slate-700 rounded-xl border border-white/5 p-4 focus:ring-2 focus:ring-purple-500/50 outline-none text-sm transition-all font-bold"
                />
            </div>

            <div>
                <label className="block text-[10px] text-purple-400 font-black uppercase tracking-widest mb-2">당신은 어떤 사람인가요?</label>
                <textarea 
                    value={background}
                    onChange={(e) => setBackground(e.target.value)}
                    placeholder="예: 저는 5년차 IT 솔루션 영업맨입니다. 대형 엔터프라이즈 고객을 주로 상대합니다."
                    className="w-full h-24 bg-slate-950 text-white placeholder-slate-700 rounded-xl border border-white/5 p-4 focus:ring-2 focus:ring-purple-500/50 outline-none resize-none text-sm transition-all"
                />
            </div>

            <div>
                <label className="block text-[10px] text-purple-400 font-black uppercase tracking-widest mb-2">분석 및 진단 스타일 (목표)</label>
                <textarea 
                    value={goal}
                    onChange={(e) => setGoal(e.target.value)}
                    placeholder="예: 빠른 상황 파악 후 즉각적인 방향성을 제시하는 스타일입니다. 고객의 고통(Pain Point)을 확대시키는 것이 핵심입니다."
                    className="w-full h-24 bg-slate-950 text-white placeholder-slate-700 rounded-xl border border-white/5 p-4 focus:ring-2 focus:ring-purple-500/50 outline-none resize-none text-sm transition-all"
                />
            </div>

            <div className="flex items-center justify-between p-4 bg-slate-950/50 rounded-xl border border-white/5">
                <div className="flex flex-col">
                    <span className="text-sm font-bold text-white">분석 반영 활성화</span>
                    <span className="text-[10px] text-slate-500">모든 분석 결과에 내 프로필을 반영합니다.</span>
                </div>
                <button 
                    onClick={() => setIsActive(!isActive)}
                    className={`w-12 h-6 rounded-full transition-all relative ${isActive ? 'bg-purple-600' : 'bg-slate-800'}`}
                >
                    <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${isActive ? 'left-7' : 'left-1'}`}></div>
                </button>
            </div>
        </div>

        <div className="mt-8 flex gap-3">
            <button 
                onClick={handleSave}
                className="flex-1 py-3 bg-purple-600 hover:bg-purple-500 text-white rounded-xl font-black transition-all shadow-xl shadow-purple-900/20 active:scale-95 text-sm"
            >
                프로필 저장하기
            </button>
            <button 
                onClick={onClose}
                className="px-6 py-3 bg-slate-800 text-slate-300 rounded-xl font-bold transition-all border border-white/5 hover:bg-slate-700 text-sm"
            >
                취소
            </button>
        </div>
      </div>
    </div>
  );
};
