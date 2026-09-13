#!/bin/bash
# 一键跑"回答质量评测"（需完整环境：Neo4j + MySQL + Milvus + Redis + kb_ingest）
# 用法：bash scripts/run_answer_quality.sh
set -e
cd "$(dirname "$0")/.."

echo "=== ① 全量端到端评测（拿 91 条知识类样本的真实回答）==="
/d/Anaconda/Anaconda3_2020_11/envs/gustobot/python -m eval.run_e2e_eval \
    --output eval/data/e2e_eval_report.json

echo
echo "=== ② 回答质量评测（LLM-as-Judge 三维分 + 关键事实覆盖率 + 引用一致性）==="
/d/Anaconda/Anaconda3_2020_11/envs/gustobot/python -m eval.run_answer_quality \
    --details eval/data/e2e_eval_report_details.jsonl \
    --output eval/data/answer_quality_report.json

echo
echo "=== 回答质量报告 ==="
cat eval/data/answer_quality_report.json
