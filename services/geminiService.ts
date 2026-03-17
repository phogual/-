
import { GoogleGenAI, Type, GenerateContentResponse } from "@google/genai";
import { AnalysisResult, ChatMessage, PreMeetingStrategy, UserPersona } from "../types";

const MODEL_NAME_PRO = 'gemini-3-flash-preview';
const MODEL_NAME_FLASH = 'gemini-3-flash-preview';

// API Key resolution: Prefer GEMINI_API_KEY for free models, fallback to API_KEY
const getApiKey = () => process.env.GEMINI_API_KEY || process.env.API_KEY || "";

/**
 * AI 응답 텍스트에서 순수 JSON만 추출하는 헬퍼 함수
 */
function extractJson(text: string): string {
  try {
    const jsonMatch = text.match(/\{[\s\S]*\}|\[[\s\S]*\]/);
    return jsonMatch ? jsonMatch[0] : text;
  } catch (e) {
    return text;
  }
}

async function generateContentWithRetry(
  ai: GoogleGenAI, 
  params: any, 
  onProgress?: (m: string) => void,
  retryCount = 0
): Promise<GenerateContentResponse> {
  try {
    return await ai.models.generateContent(params);
  } catch (err: any) {
    const errorText = String(err.message || err).toLowerCase();
    const statusCode = err.status || err.code || 0;
    
    if (statusCode === 429 || statusCode === 503 || errorText.includes("quota") || errorText.includes("overloaded")) {
      if (retryCount < 3) {
        const waitTime = Math.pow(2, retryCount) * 2000;
        onProgress?.(`서비 부하로 재시도 중... (${retryCount + 1}/3)`);
        await new Promise(resolve => setTimeout(resolve, waitTime));
        return generateContentWithRetry(ai, params, onProgress, retryCount + 1);
      }
      if (params.model === MODEL_NAME_PRO) {
        onProgress?.("고속 분석 엔진으로 자동 전환합니다...");
        return generateContentWithRetry(ai, { ...params, model: MODEL_NAME_FLASH }, onProgress, 0);
      }
    }
    throw err;
  }
}

const STRICT_GROUNDING_INSTRUCTION = `
[절대 준수 사항: 데이터 무결성 및 피드백 강도]
1. 당신은 오직 사용자가 제공한 텍스트/파일 데이터에만 기반하여 응답해야 합니다.
2. 소스에 없는 성공 사례, 특정 인물 스토리, 나이대 언급을 단 하나라도 지어내지 마십시오.
3. "대표님은 할 수 있습니다", "충분히 잘하고 계십니다"와 같은 근거 없는 응원이나 완곡한 표현은 **절대 금지**입니다.
4. 상담자의 무능함, 시스템적 결함, 놓친 기회비용을 **'뼈 때리는' 수준으로 날카롭고 직설적으로** 지적하십시오. 
5. 피드백은 비즈니스적 관점에서 '이대로 가면 망한다'는 위기감을 줄 수 있을 만큼 냉철해야 합니다.
`;

const getSystemInstruction = (persona?: UserPersona) => {
  return `
당신은 세계 최고의 세일즈 전략가이자, 타협 없는 냉철한 비즈니스 코치입니다. 
${STRICT_GROUNDING_INSTRUCTION}
사용자 페르소나(${persona?.name || '전문가'})의 톤앤매너를 유지하되, 모든 분석은 팩트 중심의 '독설적 통찰'이어야 합니다.

[핵심 지침: SPIN 기반 통합 분석 및 한국어 출력]
1. 언어 제한: 모든 분석 내용, 질문, 스크립트, 요약은 반드시 **한국어**로만 작성하십시오.
2. SPIN 통합 로직: 대화 전문에서 상담자가 실제로 던진 모든 SPIN 질문을 누락 없이 추출하십시오.
   - **Situation (상황)**: 고객의 현재 데이터와 정체성을 파악하는 질문 (최소 4개 추출)
   - **Problem (문제)**: 시스템적 병목과 심층 고통(Deep Pain)을 드러내는 질문 (최소 4개 추출)
   - **Implication (시사)**: 문제 방치 시의 기회비용과 손실 회피를 시각화하는 질문 (최소 4개 추출)
   - **Need-Payoff (해결)**: 고객 스스로 해결책의 가치를 선언하게 유도하는 질문 (최소 4개 추출)

3. 질문 추출 및 정밀 분석 원칙:
   - 상담자가 실제로 한 질문 원문(original)을 누락 없이 추출하십시오.
   - 각 질문은 반드시 **한국어**로만 작성하십시오.

4. 출력 섹션별 상세 지침:
   - 'summary': 미팅의 실패 원인이나 상담자의 치명적인 실수를 중심으로 종합 진단하십시오. (2~3문장)
   - 'consultantFeedback': 
     - 'strengths': 데이터상으로 확실히 증명된 상담자의 강점을 최소 3~4가지 구체적으로 나열하십시오.
     - 'improvements': **이 리포트의 핵심입니다.** 상담자가 놓친 심리적 트리거, 시스템적 부재, 고객에게 주도권을 뺏긴 순간 등을 아주 날카롭고 직설적으로 지적하십시오. "이 부분을 고치지 않으면 계약은 불가능하다"는 수준의 조언이 필요합니다.
   - 'spinAnalysis': 각 단계별 상담자의 질문 전략 중 **치명적으로 잘못된 점(Wrong Point)**과 이를 즉시 개선하기 위한 **날카로운 조언(Advice)**을 1~2문장으로 정의하십시오. (예: "상황 질문이 너무 많아 고객이 지루해했습니다. 즉시 문제 질문으로 넘어가 고통을 건드렸어야 했습니다.")

5. 성장 포인트 및 권장 스크립트 (이미지 기반 피드백):
   - 'growthPoints': 상담의 결정적 패착이나 전략적 보완점을 2개 추출하십시오.
     - 예: "성장 포인트 1: 상담 초반 상황 파악에 너무 긴 시간을 썼습니다. 매출 데이터가 있다면 시사 질문으로 즉시 넘어가야 했습니다."
     - 예: "전략적 보완점 2: 해결책 제시 단계에서 시스템 설명이 너무 길어 고객이 정보 과부하를 느꼈습니다."
   - 'recommendedScripts': 마스터급 상담을 위한 구체적인 권장 화법을 2개 제시하십시오.
     - 예: "고객이 '가족에 대한 책임감'을 언급했을 때, 이를 해결했을 때의 감정적 보상을 더 극적으로 시각화하십시오."
     - 예: "기능적 설명보다는 그 과정을 통과한 후 고객이 어떤 존재로 변해 있을지에 대한 '정체성 변화'에 집중하십시오."

6. 사전 전략(Pre-Meeting) 구성:
   - 고객의 고통을 극대화하고, 당신의 솔루션이 유일한 '가교(Bridge)'임을 증명하는 공격적인 전략을 수립하십시오.
`;
};

const INSIGHTS_SCHEMA = {
  charlieMorganInsight: {
    type: Type.OBJECT,
    properties: { deepPain: { type: Type.STRING }, gapDefinition: { type: Type.STRING }, bridgePositioning: { type: Type.STRING }, objectionStrategy: { type: Type.STRING } },
    required: ["deepPain", "gapDefinition", "bridgePositioning", "objectionStrategy"]
  },
  cialdiniInsight: {
    type: Type.OBJECT,
    properties: {
      preSuasionStrategy: { type: Type.STRING },
      framingLogic: { type: Type.STRING },
      structuredQuestions: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: { principle: { type: Type.STRING }, intent: { type: Type.STRING }, question: { type: Type.STRING } },
          required: ["principle", "intent", "question"]
        }
      }
    },
    required: ["preSuasionStrategy", "framingLogic", "structuredQuestions"]
  }
};

const STRATEGY_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    clientContext: { type: Type.STRING },
    strategySummary: { type: Type.STRING },
    charlieMorganInsight: INSIGHTS_SCHEMA.charlieMorganInsight,
    cialdiniInsight: INSIGHTS_SCHEMA.cialdiniInsight,
    spinQuestions: { 
        type: Type.OBJECT, 
        properties: { 
            situation: { type: Type.ARRAY, items: { type: Type.STRING } }, 
            problem: { type: Type.ARRAY, items: { type: Type.STRING } }, 
            implication: { type: Type.ARRAY, items: { type: Type.STRING } }, 
            needPayoff: { type: Type.ARRAY, items: { type: Type.STRING } } 
        },
        required: ["situation", "problem", "implication", "needPayoff"]
    },
    spinScores: {
        type: Type.OBJECT,
        properties: {
            situation: { type: Type.NUMBER },
            problem: { type: Type.NUMBER },
            implication: { type: Type.NUMBER },
            needPayoff: { type: Type.NUMBER }
        },
        required: ["situation", "problem", "implication", "needPayoff"]
    },
    spinAnalysis: {
        type: Type.OBJECT,
        properties: {
            situation: { type: Type.STRING },
            problem: { type: Type.STRING },
            implication: { type: Type.STRING },
            needPayoff: { type: Type.STRING }
        },
        required: ["situation", "problem", "implication", "needPayoff"]
    },
    persuasionStrategies: { 
        type: Type.ARRAY, 
        items: { 
            type: Type.OBJECT, 
            properties: { principle: { type: Type.STRING }, description: { type: Type.STRING }, script: { type: Type.STRING } },
            required: ["principle", "description", "script"]
        } 
    },
    tips: { type: Type.ARRAY, items: { type: Type.STRING } }
  },
  required: ["clientContext", "strategySummary", "charlieMorganInsight", "cialdiniInsight", "spinQuestions", "spinScores", "spinAnalysis", "persuasionStrategies", "tips"]
};

const SPIN_QUESTION_ITEM_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    original: { type: Type.STRING, description: "상담자가 실제로 한 질문 (태그 포함)" },
    betterVersion: { type: Type.STRING, description: "더 나은 질문 제안 또는 개선된 표현" }
  },
  required: ["original", "betterVersion"]
};

const ANALYSIS_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    contactInfo: { type: Type.OBJECT, properties: { name: { type: Type.STRING } }, required: ["name"] },
    summary: { type: Type.STRING },
    consultantFeedback: {
        type: Type.OBJECT,
        properties: {
            strengths: { type: Type.STRING },
            improvements: { type: Type.STRING }
        },
        required: ["strengths", "improvements"]
    },
    spinScore: { type: Type.INTEGER },
    spinCounts: { type: Type.OBJECT, properties: { situation: { type: Type.INTEGER }, problem: { type: Type.INTEGER }, implication: { type: Type.INTEGER }, needPayoff: { type: Type.INTEGER } }, required: ["situation", "problem", "implication", "needPayoff"] },
    spinQuestions: { 
        type: Type.OBJECT, 
        properties: { 
            situation: { type: Type.ARRAY, items: SPIN_QUESTION_ITEM_SCHEMA }, 
            problem: { type: Type.ARRAY, items: SPIN_QUESTION_ITEM_SCHEMA }, 
            implication: { type: Type.ARRAY, items: SPIN_QUESTION_ITEM_SCHEMA }, 
            needPayoff: { type: Type.ARRAY, items: SPIN_QUESTION_ITEM_SCHEMA } 
        }, 
        required: ["situation", "problem", "implication", "needPayoff"] 
    },
    spinScores: { type: Type.OBJECT, properties: { situation: { type: Type.NUMBER }, problem: { type: Type.NUMBER }, implication: { type: Type.NUMBER }, needPayoff: { type: Type.NUMBER } }, required: ["situation", "problem", "implication", "needPayoff"] },
    spinAnalysis: {
        type: Type.OBJECT,
        properties: {
            situation: { type: Type.STRING },
            problem: { type: Type.STRING },
            implication: { type: Type.STRING },
            needPayoff: { type: Type.STRING }
        },
        required: ["situation", "problem", "implication", "needPayoff"]
    },
    influenceAnalysis: { type: Type.OBJECT, properties: { reciprocity: { type: Type.INTEGER }, socialProof: { type: Type.INTEGER }, authority: { type: Type.INTEGER }, consistency: { type: Type.INTEGER }, liking: { type: Type.INTEGER }, scarcity: { type: Type.INTEGER } }, required: ["reciprocity", "socialProof", "authority", "consistency", "liking", "scarcity"] },
    persuasionAudit: { type: Type.ARRAY, items: { type: Type.OBJECT, properties: { principle: { type: Type.STRING }, detectedAction: { type: Type.STRING }, improvement: { type: Type.STRING }, score: { type: Type.INTEGER } }, required: ["principle", "detectedAction", "improvement", "score"] } },
    charlieMorganInsight: INSIGHTS_SCHEMA.charlieMorganInsight,
    cialdiniInsight: INSIGHTS_SCHEMA.cialdiniInsight,
    strengths: { type: Type.ARRAY, items: { type: Type.STRING } },
    keyMistakes: { type: Type.ARRAY, items: { type: Type.STRING } },
    betterApproaches: { type: Type.ARRAY, items: { type: Type.STRING } },
    growthPoints: {
        type: Type.ARRAY,
        items: {
            type: Type.OBJECT,
            properties: {
                title: { type: Type.STRING },
                description: { type: Type.STRING }
            },
            required: ["title", "description"]
        }
    },
    recommendedScripts: {
        type: Type.ARRAY,
        items: {
            type: Type.OBJECT,
            properties: {
                title: { type: Type.STRING },
                script: { type: Type.STRING }
            },
            required: ["title", "script"]
        }
    }
  },
  required: ["contactInfo", "summary", "consultantFeedback", "spinScore", "spinCounts", "spinQuestions", "spinScores", "spinAnalysis", "influenceAnalysis", "persuasionAudit", "charlieMorganInsight", "cialdiniInsight", "strengths", "keyMistakes", "betterApproaches", "growthPoints", "recommendedScripts"]
};

/**
 * 파일의 MIME 타입을 안전하게 가져오거나 확장자로 추측합니다.
 */
function getMimeType(file: File): string {
  if (file.type) return file.type;
  const name = file.name.toLowerCase();
  if (name.endsWith('.pdf')) return 'application/pdf';
  if (name.endsWith('.txt')) return 'text/plain';
  if (name.endsWith('.docx')) return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
  if (name.endsWith('.mp3')) return 'audio/mpeg';
  if (name.endsWith('.wav')) return 'audio/wav';
  if (name.endsWith('.m4a')) return 'audio/mp4';
  if (name.endsWith('.mp4')) return 'video/mp4';
  return 'application/octet-stream';
}

/**
 * 파일을 Base64로 변환하는 프로미스 (에러 핸들링 포함)
 */
async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const base64 = result.split(',')[1];
      if (base64) resolve(base64);
      else reject(new Error("파일 변환에 실패했습니다. (Base64 empty)"));
    };
    reader.onerror = () => reject(new Error("파일을 읽는 중 오류가 발생했습니다."));
    reader.onabort = () => reject(new Error("파일 읽기가 중단되었습니다."));
    reader.readAsDataURL(file);
    
    // 타임아웃 처리 (30초)
    setTimeout(() => reject(new Error("파일 읽기 시간이 초과되었습니다.")), 30000);
  });
}

export const analyzeSalesFile = async (file: File, persona?: UserPersona, onProgress?: (m: string) => void): Promise<AnalysisResult> => {
    const ai = new GoogleGenAI({ apiKey: getApiKey() });
    const base64 = await fileToBase64(file);
    const mimeType = getMimeType(file);
    
    onProgress?.("팩트 기반 정밀 데이터 분석 중...");
    const response = await generateContentWithRetry(ai, {
        model: MODEL_NAME_PRO,
        contents: { parts: [{ inlineData: { mimeType, data: base64 } }, { text: "세일즈 대화를 분석하십시오. 소스에 없는 거짓 정보는 배제하십시오." }] },
        config: { systemInstruction: getSystemInstruction(persona), responseMimeType: "application/json", responseSchema: ANALYSIS_SCHEMA, thinkingConfig: { thinkingLevel: 'LOW' } } as any
    }, onProgress);
    return JSON.parse(extractJson(response.text || "{}"));
};

export const analyzeSalesText = async (input: string | File, persona?: UserPersona, onProgress?: (m: string) => void): Promise<AnalysisResult> => {
    const ai = new GoogleGenAI({ apiKey: getApiKey() });
    const parts: any[] = [];
    if (input instanceof File) {
        const base64 = await fileToBase64(input);
        const mimeType = getMimeType(input);
        parts.push({ inlineData: { mimeType, data: base64 } });
    } else parts.push({ text: input });
    parts.push({ text: "텍스트 기반 진단 리포트를 생성하십시오. 거짓 정보를 지어내지 마십시오." });
    const response = await generateContentWithRetry(ai, {
        model: MODEL_NAME_PRO,
        contents: { parts },
        config: { systemInstruction: getSystemInstruction(persona), responseMimeType: "application/json", responseSchema: ANALYSIS_SCHEMA, thinkingConfig: { thinkingLevel: 'LOW' } } as any
    }, onProgress);
    return JSON.parse(extractJson(response.text || "{}"));
};

export const generatePreMeetingStrategy = async (context: string | File, persona?: UserPersona, onProgress?: (m: string) => void): Promise<PreMeetingStrategy> => {
    const ai = new GoogleGenAI({ apiKey: getApiKey() });
    const parts: any[] = [];
    if (context instanceof File) {
        const base64 = await fileToBase64(context);
        const mimeType = getMimeType(context);
        parts.push({ inlineData: { mimeType, data: base64 } });
    } else parts.push({ text: `고객 상황: ${context}` });
    const response = await generateContentWithRetry(ai, {
        model: MODEL_NAME_PRO,
        contents: { parts },
        config: { systemInstruction: getSystemInstruction(persona), responseMimeType: "application/json", responseSchema: STRATEGY_SCHEMA, thinkingConfig: { thinkingLevel: 'LOW' } } as any
    }, onProgress);
    return JSON.parse(extractJson(response.text || "{}"));
};

export const chatWithSalesCoach = async (message: string, history: ChatMessage[], file?: File, persona?: UserPersona): Promise<string> => {
    const ai = new GoogleGenAI({ apiKey: getApiKey() });
    const parts: any[] = [];
    if (file) {
        const base64 = await fileToBase64(file);
        const mimeType = getMimeType(file);
        parts.push({ inlineData: { mimeType, data: base64 } });
    }
    parts.push({ text: message });
    const response = await generateContentWithRetry(ai, {
        model: MODEL_NAME_PRO,
        contents: { parts },
        config: { systemInstruction: getSystemInstruction(persona), thinkingConfig: { thinkingLevel: 'LOW' } }
    });
    return response.text || "";
};
