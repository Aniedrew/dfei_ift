# 对话 Log 备份

本目录备份了 2026-08-25 至今的 Trae AI 对话记录（`~/.trae-cn` 与 `~/.trae-cn-server`）。

## 内容

- `alaudalog/<instance_id>/` — 原始对话日志（AalGA 二进制压缩格式，可按实例追溯）。
  - `ai_agent-*.alaudalog`：agent 对话记录
  - `toolhost-*.alaudalog`：工具执行记录
- `memory/` — Trae 长期记忆摘要（项目约定、用户偏好、会话主题汇总，纯文本 md/jsonl）。

## 说明

- `.alaudalog` 为客户端私有二进制格式，备份以保留原始数据为目的，不保证可离线解析。
- `~/.trae-cn-server/ai-agent/database.db`（80MB + 18GB WAL）为活动数据库，主文件已不可读，未纳入备份。
- 更早的对话（2026-08-25 之前）未在本地保留。
