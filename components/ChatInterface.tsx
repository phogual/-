
import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage, UserPersona } from '../types';
import { chatWithSalesCoach } from '../services/geminiService';

interface ChatInterfaceProps {
  persona?: UserPersona;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ persona }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'model',
      text: '안녕하세요. SPIN 세일즈 전문 코치입니다. 세일즈 전략이나 질문법에 대해 무엇이든 물어보세요. 녹음 파일이나 스크립트를 첨부하여 분석을 요청하실 수도 있습니다.',
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [attachedFile, setAttachedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setAttachedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setAttachedFile(e.target.files[0]);
    }
    e.target.value = '';
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if ((!input.trim() && !attachedFile) || isLoading) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      text: attachedFile ? `[파일 첨부: ${attachedFile.name}]\n${input}` : input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMsg]);
    const currentInput = input;
    const currentFile = attachedFile;
    
    setInput('');
    setAttachedFile(null);
    setIsLoading(true);

    try {
      const responseText = await chatWithSalesCoach(currentInput || "이 파일을 분석해줘.", messages, currentFile || undefined, persona);
      
      const aiMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'model',
        text: responseText,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, aiMsg]);
    } catch (error) {
      console.error(error);
      const errorMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'model',
        text: "죄송합니다. 파일을 분석하는 중 오류가 발생했습니다. 파일 크기를 확인하거나 다시 시도해주세요.",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div 
        className={`w-full max-w-4xl mx-auto h-[500px] sm:h-[600px] flex flex-col bg-slate-900/50 backdrop-blur-md rounded-2xl border transition-all duration-300 overflow-hidden shadow-2xl animate-fade-in relative ${dragActive ? 'border-indigo-500 ring-2 ring-indigo-500/20 bg-slate-900/80' : 'border-white/10'}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
    >
      {/* Drag Overlay */}
      {dragActive && (
          <div className="absolute inset-0 z-50 bg-indigo-600/10 backdrop-blur-[2px] flex flex-col items-center justify-center pointer-events-none">
              <div className="bg-slate-900 p-8 rounded-3xl border-2 border-dashed border-indigo-500 shadow-2xl animate-bounce">
                  <svg className="w-12 h-12 text-indigo-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                  <p className="text-white font-black text-lg">파일을 여기에 놓으세요</p>
              </div>
          </div>
      )}

      {/* Header */}
      <div className="p-4 border-b border-white/5 bg-slate-950/50 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse"></div>
          <h3 className="text-indigo-200 font-bold tracking-wide text-xs sm:text-sm uppercase">SPIN Strategy Coach</h3>
          {persona?.isActive && (
            <span className="text-[9px] bg-purple-500/20 text-purple-400 px-2 py-0.5 rounded border border-purple-500/30 font-bold">PERSONA ACTIVE</span>
          )}
        </div>
        <div className="text-[10px] text-slate-500 font-mono tracking-widest">MULTIMODAL AI</div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6 custom-scrollbar relative z-10">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] sm:max-w-[80%] rounded-2xl px-4 py-3 sm:px-5 sm:py-4 text-sm leading-relaxed shadow-lg transition-all duration-300 ${
                msg.role === 'user'
                  ? 'bg-gradient-to-br from-indigo-600 to-indigo-700 text-white rounded-tr-none border border-indigo-500/30'
                  : 'bg-slate-800/80 text-slate-200 rounded-tl-none border border-white/5'
              }`}
            >
              {msg.role === 'model' && (
                <div className="flex items-center gap-2 mb-2 border-b border-white/5 pb-2">
                   <div className="w-4 h-4 rounded bg-cyan-900/50 flex items-center justify-center border border-cyan-500/30 text-[8px] font-bold text-cyan-400">AI</div>
                   <span className="text-[10px] text-cyan-400 font-bold uppercase tracking-wider">SPIN Coach</span>
                </div>
              )}
              <div className="whitespace-pre-wrap font-light tracking-tight">{msg.text}</div>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-slate-800/50 rounded-2xl rounded-tl-none px-5 py-4 border border-white/5 flex gap-1.5 items-center">
              <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce"></span>
              <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce delay-150"></span>
              <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce delay-300"></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-3 sm:p-4 bg-slate-950/80 border-t border-white/5 relative z-20">
        {/* Attached File Preview */}
        {attachedFile && (
            <div className="mb-3 animate-fade-in">
                <div className="inline-flex items-center gap-2 bg-indigo-500/10 border border-indigo-500/30 px-3 py-1.5 rounded-full">
                    <svg className="w-3.5 h-3.5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"></path></svg>
                    <span className="text-[11px] text-white font-bold truncate max-w-[200px]">{attachedFile.name}</span>
                    <button onClick={() => setAttachedFile(null)} className="text-indigo-400 hover:text-white transition-colors ml-1">
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd"></path></svg>
                    </button>
                </div>
            </div>
        )}

        <form onSubmit={handleSubmit} className="flex items-center gap-3">
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            className="hidden" 
            accept="audio/*,video/*,application/pdf,text/plain"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="p-3 bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-indigo-400 rounded-xl border border-white/5 transition-all active:scale-95 shrink-0"
            title="파일 첨부"
            disabled={isLoading}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"></path></svg>
          </button>
          
          <div className="relative flex-1">
            <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={attachedFile ? "파일에 대해 궁금한 점을 입력하세요..." : "세일즈 전략에 대해 물어보세요..."}
                className="w-full bg-slate-900/50 text-white placeholder-slate-500 rounded-xl py-3.5 pl-5 pr-14 border border-white/10 focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/50 focus:outline-none transition-all shadow-inner text-sm"
                disabled={isLoading}
            />
            <button
                type="submit"
                disabled={(!input.trim() && !attachedFile) || isLoading}
                className="absolute right-2 top-2 p-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-600 text-white rounded-lg transition-colors shadow-lg shadow-indigo-500/20 active:scale-90"
            >
                <svg className="w-5 h-5 transform rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
