
import React, { ChangeEvent, useState, useRef } from 'react';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  onScriptSelect: (input: string | File) => void;
  disabled: boolean;
}

type UploadMode = 'FILE' | 'SCRIPT';

export const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect, onScriptSelect, disabled }) => {
  const [mode, setMode] = useState<UploadMode>('SCRIPT');
  const [script, setScript] = useState('');
  const [scriptFile, setScriptFile] = useState<File | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  
  const scriptFileInputRef = useRef<HTMLInputElement>(null);
  const mediaFileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  
  const MAX_FILE_SIZE = 20 * 1024 * 1024; 

  const handleFileChange = (file: File) => {
    if (file.size > MAX_FILE_SIZE) {
      alert(`파일 크기가 너무 큽니다. 웹 분석의 안정성을 위해 20MB 이하의 파일을 이용해 주세요.`);
      return;
    }
    
    const isDoc = ['.pdf', '.txt', '.docx'].some(ext => file.name.toLowerCase().endsWith(ext));
    const isMedia = ['.mp3', '.wav', '.mp4', '.m4a'].some(ext => file.name.toLowerCase().endsWith(ext)) || file.type.startsWith('audio/') || file.type.startsWith('video/');

    if (mode === 'SCRIPT') {
        if (isDoc) setScriptFile(file);
        else alert("텍스트/PDF 탭에서는 문서 파일만 업로드 가능합니다.");
    } else {
        if (isMedia) setSelectedFile(file);
        else alert("음성/영상 탭에서는 미디어 파일만 업로드 가능합니다.");
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
      if (e.dataTransfer) {
        e.dataTransfer.dropEffect = 'copy';
      }
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileChange(e.target.files[0]);
    }
    e.target.value = '';
  };

  const handleFileSubmit = () => {
    if (selectedFile) {
      onFileSelect(selectedFile);
    }
  };

  const handleScriptSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!script.trim() && !scriptFile) return;
    onScriptSelect(scriptFile || script);
  };

  return (
    <div className="w-full flex flex-col items-center animate-fade-in px-2 sm:px-0">
      <div className="flex w-full max-w-sm p-1 bg-slate-900/50 backdrop-blur-md rounded-full border border-white/5 mb-6 relative">
        <div 
          className={`absolute h-[calc(100%-8px)] top-1 rounded-full border border-white/10 shadow-lg transition-all duration-300 ease-out ${
              mode === 'SCRIPT' ? 'left-1 w-[calc(50%-4px)] bg-purple-600/20' : 'left-[calc(50%+2px)] w-[calc(50%-4px)] bg-cyan-600/20'
          }`}
        ></div>
        <button onClick={() => setMode('SCRIPT')} className={`relative z-10 flex-1 py-3 text-xs sm:text-sm font-bold transition-colors duration-300 rounded-full ${mode === 'SCRIPT' ? 'text-purple-400' : 'text-slate-500 hover:text-slate-300'}`}>텍스트/PDF</button>
        <button onClick={() => setMode('FILE')} className={`relative z-10 flex-1 py-3 text-xs sm:text-sm font-bold transition-colors duration-300 rounded-full ${mode === 'FILE' ? 'text-cyan-400' : 'text-slate-500 hover:text-slate-300'}`}>음성/영상 파일</button>
      </div>

      <div className="w-full relative group flex flex-col">
        {mode === 'SCRIPT' ? (
          <div 
            className={`w-full flex flex-col rounded-3xl bg-slate-900/50 border transition-all ${dragActive ? 'border-purple-500 ring-2 ring-purple-500/20 bg-purple-500/5' : 'border-slate-800'}`}
            onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}
          >
             <form onSubmit={handleScriptSubmit} className="relative z-10 w-full p-4 sm:p-5 flex flex-col gap-4">
                <div className="relative">
                  <textarea 
                      placeholder="대화 내용을 입력하거나 파일을 여기에 끌어다 놓으세요..." 
                      className={`w-full min-h-[300px] bg-slate-950 text-white rounded-xl border border-white/5 p-6 focus:ring-2 focus:ring-purple-500/30 transition-all resize-none text-base leading-relaxed ${scriptFile ? 'opacity-20' : ''}`}
                      value={script} onChange={(e) => setScript(e.target.value)}
                      disabled={disabled || !!scriptFile}
                  />
                  {scriptFile && (
                      <div className="absolute inset-0 bg-slate-900/95 rounded-xl flex flex-col items-center justify-center border-2 border-purple-500/50 z-10 backdrop-blur-md p-4 text-center">
                          <p className="text-white text-base font-black mb-1 truncate max-w-full">{scriptFile.name}</p>
                          <button type="button" onClick={() => setScriptFile(null)} className="px-6 py-2 bg-slate-800 text-slate-300 rounded-lg border border-white/10 font-bold">제거</button>
                      </div>
                  )}
                </div>

                <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
                    <button 
                        type="button" 
                        onClick={() => scriptFileInputRef.current?.click()} 
                        className="w-full sm:w-auto px-5 py-3 bg-slate-800/80 hover:bg-slate-700 text-slate-200 rounded-xl border border-white/10 flex items-center justify-center gap-2.5 transition-all active:scale-95"
                        disabled={disabled || !!scriptFile}
                    >
                        <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"></path></svg>
                        <span className="font-bold text-xs text-white">문서 파일 첨부</span>
                    </button>
                    <button type="submit" disabled={disabled || (!script.trim() && !scriptFile)} className="w-full sm:w-52 py-4 bg-purple-600 hover:bg-purple-500 text-white rounded-xl transition-all font-black shadow-xl active:scale-95">진단 시작</button>
                </div>
                <input type="file" ref={scriptFileInputRef} onChange={handleInputChange} accept=".pdf,.txt,.docx" className="hidden" />
             </form>
          </div>
        ) : (
          <div className="w-full flex-1 min-h-[350px] flex flex-col">
            <label
              onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}
              className={`flex flex-col items-center justify-center w-full flex-1 rounded-3xl cursor-pointer transition-all border-2 border-dashed p-6 ${dragActive ? 'border-cyan-500 bg-cyan-500/10' : 'border-slate-800 bg-slate-900/30 hover:border-cyan-500/50'}`}
            >
              {!selectedFile ? (
                <div className="flex flex-col items-center justify-center text-center">
                  <svg className="w-16 h-16 text-slate-500 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>
                  <p className="mb-1 text-lg font-bold text-white">음성/영상 업로드</p>
                  <p className="text-xs text-slate-500">파일을 드래그하여 놓으세요 (20MB 이하)</p>
                </div>
              ) : (
                <div className="relative z-10 flex flex-col items-center text-center w-full animate-fade-in">
                  <h3 className="text-white text-lg font-black mb-1 truncate max-w-full">{selectedFile.name}</h3>
                  <p className="text-slate-500 text-[10px] mb-6 uppercase tracking-widest">{(selectedFile.size / (1024 * 1024)).toFixed(1)} MB</p>
                  <div className="flex gap-4 w-full max-w-xs">
                    <button onClick={(e) => { e.preventDefault(); handleFileSubmit(); }} className="flex-1 py-4 bg-cyan-600 hover:bg-cyan-500 text-white rounded-xl font-black shadow-xl active:scale-95">분석 시작</button>
                    <button onClick={(e) => { e.preventDefault(); setSelectedFile(null); }} className="px-6 py-4 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl font-bold border border-white/10 active:scale-95">교체</button>
                  </div>
                </div>
              )}
              <input type="file" className="hidden" accept="audio/*,video/*" onChange={handleInputChange} disabled={disabled} />
            </label>
          </div>
        )}
      </div>
    </div>
  );
};
