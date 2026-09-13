// 轻量 markdown 渲染：标题 / 加粗 / 斜体 / 行内代码 / 有序无序列表 / 链接。
// 先做 HTML 转义，避免模型或用户内容被当成标签执行。

export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function renderMarkdown(text: string): string {
  if (!text) return "";
  let s = escapeHtml(text);

  // 行内代码先抽出占位，避免其中的 * 被当作加粗
  const codes: string[] = [];
  s = s.replace(/`([^`]+)`/g, (_m, c: string) => {
    codes.push(c);
    return "\u0000" + (codes.length - 1) + "\u0000";
  });
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");

  // 生成的 <a> 先存占位、最后统一还原：否则裸链接规则会再扫一遍刚生成的 href，
  // 产出嵌套 <a>（实测 "[搜狐](url)" 会变成 <a href="<a href=...">搜狐</a>）。
  const links: string[] = [];
  const link = (url: string, label: string) => {
    links.push(
      '<a href="' + url + '" target="_blank" rel="noopener noreferrer">' + (label || url) + "</a>"
    );
    return "\u0001" + (links.length - 1) + "\u0001";
  };

  // markdown 链接 [文字](url)。必须先于"裸链接"处理，否则 URL 会被裸链接规则先吃掉。
  s = s.replace(/\[([^\]]*)\]\((https?:\/\/[^\s)]+)\)/g, (_m, label: string, url: string) =>
    link(url, label)
  );

  // 裸链接：排除括号内的内容，且尾部标点不能进 href。
  // 旧实现用 [^\s<]+，会把 "[搜狐](url)" 结尾的 ")" 一并吞进 href，
  // 于是 href 变成 "...122572149)"，链接点开是空页。
  s = s.replace(/https?:\/\/[^\s<>()\[\]（）【】「」]+/g, (m: string) => {
    const url = m.replace(/[。，、；：！？,;:!?'"”’）】」]+$/, "");
    return link(url, url) + m.slice(url.length);
  });

  s = s.replace(/\u0001(\d+)\u0001/g, (_m, i: string) => links[Number(i)]);
  s = s.replace(/\u0000(\d+)\u0000/g, (_m, i: string) => "<code>" + codes[Number(i)] + "</code>");

  const out: string[] = [];
  let listType: "ul" | "ol" | "" = "";
  let para: string[] = [];
  let blankPending = false;

  // 段落内的单换行用 <br>，但块级元素（标题/列表/段落）之间绝不插 <br>。
  const flushPara = () => {
    if (para.length) {
      out.push("<p>" + para.join("<br>") + "</p>");
      para = [];
    }
  };
  const closeList = () => {
    if (listType) {
      out.push("</" + listType + ">");
      listType = "";
    }
  };

  for (const raw of s.split("\n")) {
    const line = raw.trim();

    if (!line) {
      // 空行只做标记、不立刻关列表：否则一个列表会被切成多个 ol，序号全变 1
      blankPending = true;
      flushPara();
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      closeList();
      flushPara();
      const level = Math.min(heading[1].length + 1, 6);
      out.push("<h" + level + ">" + heading[2] + "</h" + level + ">");
      blankPending = false;
      continue;
    }

    const ul = line.match(/^[-*+]\s+(.*)$/);
    if (ul) {
      flushPara();
      if (listType !== "ul") {
        closeList();
        out.push("<ul>");
        listType = "ul";
      }
      out.push("<li>" + ul[1] + "</li>");
      blankPending = false;
      continue;
    }

    const ol = line.match(/^\d+[.)]\s+(.*)$/);
    if (ol) {
      flushPara();
      if (listType !== "ol") {
        closeList();
        out.push("<ol>");
        listType = "ol";
      }
      out.push("<li>" + ol[1] + "</li>");
      blankPending = false;
      continue;
    }

    closeList();
    para.push(line);
    blankPending = false;
  }
  flushPara();
  closeList();

  return out.join("\n");
}

/** 把一条 source 格式化成可读文本（不同来源的字段名不统一）。 */
export function formatSource(src: Record<string, unknown> | string): string {
  if (typeof src === "string") return src;
  const label =
    (src.title as string) ||
    (src.name as string) ||
    (src.source as string) ||
    (src.document_id as string) ||
    "";
  const text = label || (src.url as string) || "来源";
  return text.length > 42 ? text.slice(0, 40) + "…" : text;
}
