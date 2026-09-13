<template>
  <section class="chat-panel">
    <header class="chat-topbar">
      <button class="icon-btn sidebar-toggle" type="button" title="展开/收起会话列表" @click="state.sidebarOpen = !state.sidebarOpen">
        ☰
      </button>
      <div class="session-title">{{ displayTitle }}</div>
      <div class="topbar-right">
        <span v-if="state.isTyping" class="gen-status">生成中…</span>
      </div>
    </header>

    <div ref="scrollArea" class="chat-scroll">
      <div v-if="state.historyLoading" class="loading-hint">正在加载历史消息…</div>

      <div v-else-if="state.messages.length === 0" class="welcome">
        <div class="welcome-logo">🍳</div>
        <h1>有什么可以帮你的？</h1>
        <p>我可以查菜谱、讲饮食典故、分析你上传的文件。</p>
        <div class="suggestions">
          <button
            v-for="(item, idx) in suggestions"
            :key="idx"
            type="button"
            @click="sendSuggestion(item.prompt)"
          >
            {{ item.label }}
          </button>
        </div>
      </div>

      <div v-else class="messages">
        <div v-for="(m, index) in state.messages" :key="index" :class="['msg', m.role]">
          <div class="avatar">{{ m.role === "user" ? "我" : "G" }}</div>
          <div class="msg-body">
            <div v-if="!m.content && state.isTyping" class="typing">
              <span></span><span></span><span></span>
            </div>
            <div v-else class="msg-content" v-html="renderMarkdown(m.content)"></div>

            <div v-if="m.sources && m.sources.length" class="sources">
              <span class="sources-title">参考来源</span>
              <span v-for="(s, sIdx) in m.sources.slice(0, 4)" :key="sIdx" class="source-chip">
                {{ formatSource(s) }}
              </span>
            </div>

            <!-- 调用链路：作为注释展示，不影响回答内容 -->
            <div v-if="chainText(m)" class="chain-note">{{ chainText(m) }}</div>
          </div>
        </div>

      </div>
    </div>

    <footer class="composer">
      <div v-if="state.attachedFileName || state.uploadStatus" class="attach-row">
        <span class="attach-chip">📎 {{ state.attachedFileName || state.uploadStatus }}</span>
        <button v-if="state.attachedFileName" type="button" class="attach-clear" @click="clearAttachment">移除</button>
      </div>

      <div class="composer-box">
        <label class="attach-btn" title="上传文件">
          <input type="file" class="hidden" @change="onFileSelected" />
          <span>＋</span>
        </label>
        <textarea
          v-model="state.userInput"
          :placeholder="placeholder"
          :disabled="state.isTyping"
          rows="1"
          @keydown.enter.exact.prevent="sendMessage"
        ></textarea>
        <button
          class="send-btn"
          type="button"
          :disabled="state.isTyping || !state.userInput.trim()"
          @click="sendMessage"
        >
          ↑
        </button>
      </div>
      <p class="composer-hint">Enter 发送 · Shift + Enter 换行</p>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import { useChat } from "../composables/useChat";
import type { ChatMessage } from "../composables/useChat";
import { formatSource, renderMarkdown } from "../utils/markdown";

const { state, displayTitle, sendMessage, sendSuggestion, onFileSelected } = useChat();

const scrollArea = ref<HTMLElement | null>(null);

const suggestions = [
  { label: "推荐经典鲁菜", prompt: "请推荐几道经典鲁菜，并分别说明它们的特色。" },
  { label: "佛跳墙的典故", prompt: "佛跳墙这道菜的历史典故是什么？" },
  { label: "古人的饮茶习惯", prompt: "古代人是怎么煮茶、喝茶的？" },
  { label: "分析我上传的文件", prompt: "请帮我总结一下我上传文件里的亮点。" }
];

const placeholder = computed(() => {
  if (state.isTyping) return "正在生成回复…";
  if (state.attachedFileName) return `文件已上传：${state.attachedFileName}，请输入问题`;
  return "给 GustoBot 发送消息…";
});

// 把后端回传的调用链路拼成一行注释文字
function chainText(m: ChatMessage): string {
  const c = m.chain;
  const route = c?.route || m.route;
  const parts: string[] = [];
  if (route) parts.push("路由 " + route);
  if (c?.steps?.length) parts.push("步骤 " + c.steps.join(" → "));
  if (typeof c?.confidence === "number") parts.push("置信度 " + c.confidence.toFixed(2));
  if (c?.hallucination) parts.push("自检 " + c.hallucination);
  const n = c?.sources ?? m.sources?.length ?? 0;
  if (n) parts.push("来源 " + n);
  return parts.join("   ·   ");
}

function clearAttachment(): void {
  state.attachedFilePath = "";
  state.attachedFileName = "";
  state.uploadStatus = "";
}

function scrollToBottom(): void {
  nextTick(() => {
    const el = scrollArea.value;
    if (el) el.scrollTop = el.scrollHeight;
  });
}

// 流式期间内容在增长，用内容长度驱动自动滚动
watch(
  () => state.messages.length + (state.messages[state.messages.length - 1]?.content.length || 0),
  scrollToBottom
);
watch(() => state.activeSessionId, scrollToBottom);
</script>
