
export interface CharlieMorganInsight {
  deepPain: string;        // 심층 고통 분석
  gapDefinition: string;   // 격차 정의
  bridgePositioning: string; // 다리 포지셔닝 메시지
  objectionStrategy: string; // 반론 돌파 마인드셋
}

export interface SalesExecutionDiagnosis {
  goodPoints: string;
  painDigging: {
    status: string;
    missingQuestions: string[];
  };
  valueAddingErrors: {
    point: string;
    reason: string;
  }[];
  mercilessImprovement: {
    scriptFix: {
      situation: string;
      original: string;
      charlieStyle: string;
    }[];
    alphaFrameGuide: string;
  };
}

export interface PreSuasionQuestion {
  principle: string; // 손실 회피, 정체성 일관성, 생존 본능, 대안 폐쇄, 희귀성 중 하나
  intent: string;    // 질문의 심리적 의도
  question: string;  // 실제 질문 스크립트
}

export interface CialdiniInsight {
  preSuasionStrategy: string; // 초전 설득 전략
  framingLogic: string;       // 프레이밍 논리
  structuredQuestions: PreSuasionQuestion[]; // 5대 원칙 기반 질문
}

export interface PersuasionTactic {
  principle: string;
  description: string;
  script: string;
}

export interface PersuasionAudit {
  principle: string;
  detectedAction: string; 
  improvement: string;    
  score: number;
}

export interface AnalysisResult {
  summary: string;
  consultantFeedback: {
    strengths: string;
    improvements: string;
  };
  spinScore: number;
  preSuasionScore: number;
  outcomes: 'Order' | 'Advance' | 'Continuation' | 'No-Sale';
  contactInfo?: {
    name: string;
    phone: string;
    date: string;
  };
  charlieMorganInsight: CharlieMorganInsight;
  executionDiagnosis: SalesExecutionDiagnosis;
  cialdiniInsight: CialdiniInsight;
  spinCounts: {
    situation: number;
    problem: number;
    implication: number;
    needPayoff: number;
  };
  spinQuestions: { 
    situation: { original: string; betterVersion: string }[];
    problem: { original: string; betterVersion: string }[];
    implication: { original: string; betterVersion: string }[];
    needPayoff: { original: string; betterVersion: string }[];
  };
  spinScores: {
    situation: number;
    problem: number;
    implication: number;
    needPayoff: number;
  };
  spinAnalysis: {
    situation: string;
    problem: string;
    implication: string;
    needPayoff: string;
  };
  influenceAnalysis: {
    reciprocity: number;
    socialProof: number;
    authority: number;
    consistency: number;
    liking: number;
    scarcity: number;
  };
  persuasionAudit: PersuasionAudit[];
  strengths: string[];
  keyMistakes: string[];
  betterApproaches: string[];
  growthPoints: {
    title: string;
    description: string;
  }[];
  recommendedScripts: {
    title: string;
    script: string;
  }[];
}

export interface PreMeetingStrategy {
    clientContext: string;
    strategySummary: string;
    contactInfo?: {
        name: string;
        phone: string;
        date: string;
    };
    charlieMorganInsight: CharlieMorganInsight;
    cialdiniInsight: CialdiniInsight;
    preSuasionQuestions: {
        principle: string;
        question: string;
        intent: string;
    }[];
    spinQuestions: {
        situation: string[];
        problem: string[];
        implication: string[];
        needPayoff: string[];
    };
    spinScores: {
        situation: number;
        problem: number;
        implication: number;
        needPayoff: number;
    };
    spinAnalysis: {
        situation: string;
        problem: string;
        implication: string;
        needPayoff: string;
    };
    persuasionStrategies: PersuasionTactic[];
    tips: string[];
}

export enum AppStatus {
  IDLE = 'IDLE',
  READING_FILE = 'READING_FILE',
  ANALYZING = 'ANALYZING',
  SUCCESS = 'SUCCESS',
  ERROR = 'ERROR',
}

export type ChatRole = 'user' | 'model';

export interface ChatMessage {
  id: string;
  role: ChatRole;
  text: string;
  timestamp: Date;
}

export interface UserPersona {
  name: string;
  background: string;
  goal: string;
  isActive: boolean;
}
