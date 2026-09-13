<template>
  <div :class="['app-shell', { 'sidebar-collapsed': !state.sidebarOpen }]">
    <aside class="sidebar">
      <div class="sidebar-head">
        <div class="brand">
          <span class="brand-logo">🍳</span>
          <span class="brand-name">GustoBot</span>
        </div>
        <button class="new-chat-btn" type="button" @click="newSession">＋ 新对话</button>
      </div>

      <div class="sidebar-body">
        <div v-if="state.sessionsLoading" class="sidebar-hint">正在加载…</div>
        <div v-else-if="state.sessions.length === 0" class="sidebar-hint">还没有历史对话</div>
        <ul v-else class="session-list">
          <li
            v-for="s in state.sessions"
            :key="s.id"
            :class="['session-item', { active: s.id === state.activeSessionId }]"
          >
            <button class="session-main" type="button" @click="switchSession(s.id)">
              <span class="session-name">{{ s.title || "未命名对话" }}</span>
              <span class="session-time">{{ formatTime(s.updated_at || s.created_at) }}</span>
            </button>
            <button class="session-del" type="button" title="删除该对话" @click.stop="deleteSession(s.id)">
              ×
            </button>
          </li>
        </ul>
      </div>

      <div class="sidebar-foot">
        <span class="foot-note">{{ state.sessions.length }} 个历史对话</span>
      </div>
    </aside>

    <main class="main-area">
      <ChatPanel />
    </main>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from "vue";
import ChatPanel from "./components/ChatPanel.vue";
import { useChat } from "./composables/useChat";

const { state, switchSession, newSession, deleteSession, formatTime, init } = useChat();

onMounted(() => {
  void init();
});
</script>
