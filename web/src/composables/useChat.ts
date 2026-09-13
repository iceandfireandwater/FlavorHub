import { computed, reactive } from "vue";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";
const SESSION_KEY = "gustobot.session_id";
const USER_ID = "default_user";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  route?: string | null;
  routeLogic?: string | null;
  sources?: Array<Record<string, unknown>>;
  /** 调用链路（后端 done 事件回传），前端只作为注释展示 */
  chain?: {
    route?: string | null;
    confidence?: number | null;
    logic?: string;
    hallucination?: string;
    steps?: string[];
    sources?: number;
  } | null;
}

export interface SessionItem {
  id: string;
  title?: string;
  created_at?: string;
  updated_at?: string;
}

function loadSessionId(): string {
  try {
    return window.localStorage.getItem(SESSION_KEY) || "";
  } catch {
    return "";
  }
}

function saveSessionId(id: string): void {
  try {
    if (id) window.localStorage.setItem(SESSION_KEY, id);
    else window.localStorage.removeItem(SESSION_KEY);
  } catch {
    /* 隐私模式 / 禁用存储时忽略 */
  }
}

// 模块级单例状态：整个应用共享一份聊天状态
const state = reactive({
  sessions: [] as SessionItem[],
  sessionsLoading: false,
  activeSessionId: loadSessionId(),
  messages: [] as ChatMessage[],
  isTyping: false,
  userInput: "",
  uploadStatus: "",
  attachedFilePath: "",
  attachedFileName: "",
  historyLoading: false,
  sidebarOpen: true
});

const displayTitle = computed(() => {
  const current = state.sessions.find((s) => s.id === state.activeSessionId);
  if (current?.title) return current.title;
  return state.activeSessionId ? "对话" : "新对话";
});

async function loadSessions(): Promise<void> {
  state.sessionsLoading = true;
  try {
    const resp = await fetch(`${API_BASE}/api/v1/sessions/?user_id=${USER_ID}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = (await resp.json()) as SessionItem[];
    state.sessions = [...data].sort((a, b) =>
      String(b.updated_at || b.created_at || "").localeCompare(String(a.updated_at || a.created_at || ""))
    );
  } catch (error) {
    console.error("加载会话列表失败", error);
  } finally {
    state.sessionsLoading = false;
  }
}

async function loadHistory(sessionId: string): Promise<void> {
  if (!sessionId) {
    state.messages = [];
    return;
  }
  state.historyLoading = true;
  try {
    const resp = await fetch(`${API_BASE}/api/v1/chat/history/${sessionId}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = (await resp.json()) as Array<{
      message_type: string;
      content: string;
      route?: string | null;
      message_metadata?: {
        sources?: Array<Record<string, unknown>>;
        chain?: ChatMessage["chain"];
      } | null;
    }>;
    state.messages = data.map((m) => ({
      role: m.message_type === "user_query" ? "user" : "assistant",
      content: m.content,
      route: m.route ?? null,
      // 刷新页面后靠落库的 message_metadata 恢复来源与调用链路，
      // 否则只有最后一条（还在内存里的那条）有标注。
      sources: m.message_metadata?.sources ?? [],
      chain: m.message_metadata?.chain ?? null
    })) as ChatMessage[];
  } catch (error) {
    console.error("加载历史消息失败", error);
    state.messages = [];
  } finally {
    state.historyLoading = false;
  }
}

async function switchSession(sessionId: string): Promise<void> {
  if (state.isTyping || sessionId === state.activeSessionId) return;
  state.activeSessionId = sessionId;
  saveSessionId(sessionId);
  await loadHistory(sessionId);
}

function newSession(): void {
  if (state.isTyping) return;
  state.activeSessionId = "";
  saveSessionId("");
  state.messages = [];
  state.uploadStatus = "";
  state.attachedFilePath = "";
  state.attachedFileName = "";
  state.userInput = "";
}

async function deleteSession(sessionId: string): Promise<void> {
  if (state.isTyping) return;

  // 硬删不可恢复：会话记录 + 全部消息 + 上下文记忆都会永久消失，删前必须二次确认
  const target = state.sessions.find((s) => s.id === sessionId);
  const label = target?.title?.trim() || "这个对话";
  const confirmed = window.confirm(
    `确定要删除「${label}」吗？

会话记录、全部消息以及上下文记忆都会被永久删除，且无法恢复。`
  );
  if (!confirmed) return;

  try {
    await fetch(`${API_BASE}/api/v1/sessions/${sessionId}`, { method: "DELETE" });
  } catch (error) {
    console.error("删除会话失败", error);
  }
  if (sessionId === state.activeSessionId) {
    newSession();
  }
  await loadSessions();
}

// 后端返回的是不带时区标记的 UTC 时间串（如 2026-09-12T15:43:03），
// 直接 new Date() 会被当成本地时间，导致整体差一个时区（东八区差 8 小时）。
function parseTime(value?: string): Date | null {
  if (!value) return null;
  let s = value.trim().replace(" ", "T");
  if (!/([Zz]|[+-]\d{2}:?\d{2})$/.test(s)) s += "Z";
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? null : d;
}

function formatTime(value?: string): string {
  const d = parseTime(value);
  if (!d) return "";
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return "刚刚";
  if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} 小时前`;
  if (diff < 86400 * 7) return `${Math.floor(diff / 86400)} 天前`;
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

async function onFileSelected(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;

  state.uploadStatus = "正在上传...";
  try {
    const formData = new FormData();
    formData.append("file", file);
    const resp = await fetch(`${API_BASE}/api/v1/upload/file`, { method: "POST", body: formData });
    const data = await resp.json();
    if (data?.success) {
      state.attachedFilePath = data.file_path;
      state.attachedFileName = data.original_name || file.name;
      state.uploadStatus = `已上传：${state.attachedFileName}`;
    } else {
      throw new Error(data?.detail || "上传失败");
    }
  } catch (error) {
    console.error("文件上传失败", error);
    state.uploadStatus = "上传失败，请重试";
  } finally {
    input.value = "";
  }
}

function sendSuggestion(prompt: string): void {
  if (state.isTyping) return;
  state.userInput = prompt;
  void sendMessage();
}

async function sendMessage(): Promise<void> {
  const message = state.userInput.trim();
  if (!message || state.isTyping) return;

  state.messages.push({ role: "user", content: message });
  state.userInput = "";
  state.isTyping = true;

  const payload: Record<string, unknown> = {
    message,
    session_id: state.activeSessionId || undefined,
    stream: true
  };
  if (state.attachedFilePath) {
    payload.file_path = state.attachedFilePath;
    payload.ingest_incremental = true;
  }

  // 先插入空的助手消息，之后把 SSE 的 token 逐个追加进去
  state.messages.push({
    role: "assistant",
    content: "",
    route: null,
    routeLogic: null,
    sources: [],
    chain: null
  });
  // 必须用索引取回响应式代理，直接改 push 前的原始引用不会触发视图更新
  const assistant = state.messages[state.messages.length - 1];

  try {
    const resp = await fetch(`${API_BASE}/api/v1/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buf = "";

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      let idx: number;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const line = raw.startsWith("data:") ? raw.slice(5).trim() : raw.trim();
        if (!line) continue;

        let evt: Record<string, any>;
        try {
          evt = JSON.parse(line);
        } catch {
          continue;
        }

        if (evt.session_id) {
          state.activeSessionId = evt.session_id;
          saveSessionId(evt.session_id);
        }

        if (evt.type === "message" && evt.content) {
          assistant.content += evt.content;
        } else if (evt.type === "replace") {
          // 后端检测到整段重放时发来的更正内容
          assistant.content = evt.content || "";
        } else if (evt.type === "metadata") {
          if (evt.route) assistant.route = evt.route;
          if (evt.metadata?.logic) assistant.routeLogic = evt.metadata.logic;
          if (evt.metadata?.sources) assistant.sources = evt.metadata.sources;
        } else if (evt.type === "done") {
          if (evt.metadata?.sources) assistant.sources = evt.metadata.sources;
          if (evt.metadata?.chain) assistant.chain = evt.metadata.chain;
        } else if (evt.type === "error") {
          assistant.content += `\n\n[出错] ${evt.content || ""}`;
        }
      }
    }

    if (!assistant.content) {
      assistant.content = "抱歉，未能获取到有效的响应。";
    }
  } catch (error: unknown) {
    console.error("聊天请求失败", error);
    if (!assistant.content) {
      assistant.content = "请求出错了，请稍后重试。";
    }
  } finally {
    state.isTyping = false;
    state.attachedFilePath = "";
    state.attachedFileName = "";
    state.uploadStatus = "";
  }

  // 标题与时间会变，新会话也会出现，刷新一次列表
  void loadSessions();
}

export function useChat() {
  return {
    state,
    displayTitle,
    loadSessions,
    loadHistory,
    switchSession,
    newSession,
    deleteSession,
    sendMessage,
    sendSuggestion,
    onFileSelected,
    formatTime,
    init: async () => {
      await loadSessions();
      if (state.activeSessionId) {
        await loadHistory(state.activeSessionId);
      }
    }
  };
}
