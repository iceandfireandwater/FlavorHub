# 爬虫 / 批量导入指南

本项目的知识库（KB）目前以“菜谱文档”为基本存储单元（`/api/v1/knowledge/recipes/*`）。
为了快速填充数据，提供两种批量导入方式：

1. **Wikipedia 爬虫导入（推荐用于快速验证检索链路）**
2. **从 `data/recipe.json` 批量导入（适合一次性灌大量菜谱）**

> 说明：下面两个工具默认都通过 **HTTP API** 写入知识库，因此请先确保后端已启动并可访问。

---

## 前置条件

- 启动服务（推荐 Docker Compose）：

```bash
docker-compose up -d
```

- 确认 API 可访问：

```bash
curl -s http://localhost:8000/docs >/dev/null && echo "OK"
```

如使用了非默认端口/域名，请在命令中通过 `--api-base-url` 指定。

---

## 方式 1：Wikipedia 爬虫导入

### 1) 仅抓取预览（不写入 KB）

```bash
python -m gustobot.crawler.cli wikipedia --query "川菜" --limit 5 --dry-run
```

### 2) 抓取并写入 KB

```bash
python -m gustobot.crawler.cli wikipedia --query "川菜" --import-kb --limit 10
```

如后端不在 `http://localhost:8000`：

```bash
python -m gustobot.crawler.cli wikipedia \
  --query "川菜" \
  --import-kb \
  --limit 10 \
  --api-base-url "http://localhost:8000"
```

> 写入策略：Wikipedia 的 `extract`（摘要纯文本）会被当作一段“步骤”存入 KB，便于检索验证。

---

## 方式 2：从 JSON 文件批量导入

项目内置的 `data/recipe.json` 是一个 **以菜名为 key 的 dict**，每条数据字段大致为：
`主食材 / 辅料 / 耗时 / 口味 / 工艺 / 做法 / 类型`。

导入脚本会将其转换为 KB API 需要的字段：
`name / category / time / ingredients / steps / tips`。

### 1) 仅转换预览（不写入 KB）

```bash
python scripts/import_recipes.py --file data/recipe.json --batch-size 100 --dry-run
```

### 2) 写入 KB（可先小批量验证）

```bash
python scripts/import_recipes.py --file data/recipe.json --batch-size 100 --limit 200
```

### 3) 全量写入（耗时较长）

```bash
python scripts/import_recipes.py --file data/recipe.json --batch-size 100
```

同样支持指定后端地址：

```bash
python scripts/import_recipes.py \
  --file data/recipe.json \
  --batch-size 100 \
  --api-base-url "http://localhost:8000"
```

---

## 常见问题

- **报错：无法连接到 `http://localhost:8000`**
  - 确认 `docker-compose up -d` 已启动
  - 或通过 `--api-base-url` 指向正确地址

- **后端返回 5xx / 写入失败**
  - 批量写入会触发向量化与入库流程，请查看后端日志：
    `docker-compose logs -f backend`

