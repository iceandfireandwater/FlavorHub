"""GustoBot 评测体系 —— 评测集生成器。

数据来源（全部为项目自带知识库，保证 ground truth 客观可追溯）：
- data/recipe.json      19,669 条菜谱（主食材/辅料/耗时/口味/工艺/做法/类型/菜系）
- data/kb/data.txt      历史文化长文（FAQ/政策类）
- 手工构造多轮记忆与负样本

生成流程：
1. 模板自动生成（recipe_search / recipe_detail / recipe_compare / stat_query）
2. LLM 辅助出题（history_faq，--use-llm 时启用；否则占位待人工补）
3. 人工脚本（multi_turn / negative）
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schema import EvalCase

random.seed(42)

# 自动推导项目根：eval/scripts/dataset.py -> 上溯两级
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_recipe_db(path: str = f"{PROJECT_ROOT}/data/recipe.json") -> Dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_kb_text(path: str = f"{PROJECT_ROOT}/data/kb/data.txt") -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_kb_paragraphs(text: str) -> List[str]:
    """按双换行切段，过滤过短段落。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [p for p in paras if len(p) >= 30]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

TIME_TO_MINUTES = {
    "十分钟": 10, "十五分钟": 15, "二十分钟": 20, "二十五分钟": 25,
    "半小时": 30, "三十分钟": 30, "四十分钟": 40, "四十五分钟": 45,
    "五十分钟": 50, "一小时": 60, "一小时半": 90, "一个半小时": 90,
    "两小时": 120, "一个多小时": 70, "1小时": 60, "十分钟左右": 10,
}

SPICE_LEVEL = {
    "麻辣": 5, "辣": 4, "香辣": 4, "微辣": 3, "酸辣": 3, "咸辣": 4,
    "中辣": 4, "重辣": 5, "清淡": 0, "原味": 0, "咸鲜": 1, "鲜香": 1,
    "酸甜": 1, "甜": 1, "香甜": 1, "咸": 1, "五香": 1, "孜然": 2,
    "咖喱": 2, "蒜香": 2, "酱香": 2, "奶香": 1, "葱香": 1, "果味": 1,
    "糟香": 1, "其他": 1, "甜味": 1, "咸甜": 1, "酸": 1, "苦": 1, "清淡微辣": 3,
}


def parse_minutes(耗时: str) -> Optional[int]:
    if not 耗时:
        return None
    if 耗时 in TIME_TO_MINUTES:
        return TIME_TO_MINUTES[耗时]
    m = re.search(r"(\d+)\s*分钟", 耗时)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*小时", 耗时)
    if m:
        return int(m.group(1)) * 60
    m = re.search(r"半小时", 耗时)
    if m:
        return 30
    return None


def spice_score(口味: str) -> Optional[int]:
    return SPICE_LEVEL.get(口味)


def first_n_ingredients(entry: dict, n: int = 3) -> List[str]:
    names = []
    for grp_key in ("主食材", "辅料"):
        grp = entry.get(grp_key, [])
        if isinstance(grp, list):
            for item in grp:
                if isinstance(item, list) and item:
                    names.append(str(item[0]))
                elif isinstance(item, str):
                    names.append(item)
    return names[:n]


def _safe_fill(template: str, **kwargs) -> str:
    """安全填充模板占位符，避免数据中的 { } 被 format 解析。"""
    for k, v in kwargs.items():
        template = template.replace("{" + k + "}", str(v))
    return template


def step_count(做法: str) -> int:
    if not 做法:
        return 0
    return len(re.findall(r"\d+[:：]", 做法))


# ---------------------------------------------------------------------------
# 问题模板
# ---------------------------------------------------------------------------

SEARCH_TEMPLATES = [
    "{dish}怎么做？",
    "{dish}的做法是什么？",
    "如何制作{dish}？",
    "{dish}怎么做好吃？",
    "给我讲讲{dish}的做法",
    "{dish}的烹饪步骤是什么？",
]

DETAIL_TEMPLATES = {
    "ingredients": [
        "{dish}需要哪些食材？",
        "做{dish}要准备什么材料？",
        "{dish}的主料有哪些？",
    ],
    "time": [
        "{dish}需要多长时间做好？",
        "做{dish}要多久？",
        "{dish}的烹饪时间大概多久？",
    ],
    "taste": [
        "{dish}是什么口味？",
        "{dish}味道怎么样？",
    ],
    "process": [
        "{dish}是用什么工艺做的？",
        "{dish}的做法工艺是什么？",
    ],
    "steps": [
        "{dish}有几个步骤？",
        "做{dish}分几步？",
    ],
}

COMPARE_TEMPLATES = {
    "spice": ["{a}和{b}哪个更辣？", "{a}和{b}哪个更辣一点？"],
    "time": ["{a}和{b}哪个做起来更快？", "做{a}和做{b}哪个更省时间？"],
}

STAT_TEMPLATES = [
    "你们有多少道{key}的菜？",
    "{key}类菜谱有多少道？",
    "一共有多少道{key}？",
    "数据库里{key}有几道菜？",
]

# ---------------------------------------------------------------------------
# 场景生成器
# ---------------------------------------------------------------------------

def gen_recipe_search(recipe_db: Dict[str, dict], n: int = 30) -> List[EvalCase]:
    """菜谱搜索/做法。"""
    names = _pick_with_coverage(recipe_db, n, key="类型")
    cases = []
    for i, dish in enumerate(names):
        entry = recipe_db[dish]
        template = random.choice(SEARCH_TEMPLATES)
        question = _safe_fill(template, dish=dish)
        steps = entry.get("做法", "")
        kw = [dish] + first_n_ingredients(entry, 2)
        cases.append(EvalCase(
            id=f"rs_{i:03d}",
            scenario="recipe_search",
            turns=[question],
            expected_route="graphrag-query",
            expected_tool="cypher_query",
            expected_args={"dish": dish},
            expected_slots={"dish": dish},
            relevant_ids=[dish],
            expected_answer_keywords=kw,
            ground_truth=steps,
            gt_source=f"recipe.json:{dish}.做法",
            note=f"步数={step_count(steps)}",
        ))
    return cases


def gen_recipe_detail(recipe_db: Dict[str, dict], n: int = 25) -> List[EvalCase]:
    """菜谱详情（食材/耗时/口味/工艺/步骤）。"""
    subtypes = ["ingredients", "time", "taste", "process", "steps"]
    per = n // len(subtypes)
    cases = []
    idx = 0
    for st in subtypes:
        names = _pick_with_coverage(recipe_db, per, key="类型")
        for dish in names:
            entry = recipe_db[dish]
            template = random.choice(DETAIL_TEMPLATES[st])
            question = _safe_fill(template, dish=dish)
            if st == "ingredients":
                gt = "；".join(
                    f"{x[0]}({x[1]})" if isinstance(x, list) and len(x) > 1 else str(x)
                    for x in (entry.get("主食材", []) + entry.get("辅料", []))[:8]
                )
                kw = first_n_ingredients(entry, 3)
            elif st == "time":
                gt = entry.get("耗时", "")
                kw = [gt] if gt else []
            elif st == "taste":
                gt = entry.get("口味", "")
                kw = [gt] if gt else []
            elif st == "process":
                gt = entry.get("工艺", "")
                kw = [gt] if gt else []
            else:
                gt = f"{step_count(entry.get('做法',''))} 步"
                kw = []
            cases.append(EvalCase(
                id=f"rd_{idx:03d}",
                scenario="recipe_detail",
                turns=[question],
                expected_route="graphrag-query",
                expected_tool="cypher_query",
                expected_args={"dish": dish, "field": st},
                expected_slots={"dish": dish},
                relevant_ids=[dish],
                expected_answer_keywords=kw,
                ground_truth=gt,
                gt_source=f"recipe.json:{dish}.{st}",
                note=st,
            ))
            idx += 1
    return cases


def gen_recipe_compare(recipe_db: Dict[str, dict], n: int = 15) -> List[EvalCase]:
    """菜谱对比（辣度 / 耗时）。"""
    names = [k for k in recipe_db if recipe_db[k].get("口味") and recipe_db[k].get("耗时")]
    cases = []
    made = 0
    guard = 0
    while made < n and guard < 500:
        guard += 1
        a, b = random.sample(names, 2)
        sa, sb = spice_score(recipe_db[a]["口味"]), spice_score(recipe_db[b]["口味"])
        ta, tb = parse_minutes(recipe_db[a]["耗时"]), parse_minutes(recipe_db[b]["耗时"])
        mode = None
        gt = ""
        if sa is not None and sb is not None and sa != sb:
            mode = "spice"
            winner = a if sa > sb else b
            gt = f"{recipe_db[winner]['口味']}，{winner}更辣"
        elif ta is not None and tb is not None and ta != tb:
            mode = "time"
            winner = a if ta < tb else b
            gt = f"{recipe_db[winner]['耗时']}，{winner}更快"
        if not mode:
            continue
        loser = b if winner == a else a
        if mode == "spice":
            gt = f"{winner}更辣（{recipe_db[winner]['口味']} vs {recipe_db[loser]['口味']}）"
        else:
            gt = f"{winner}更快（{recipe_db[winner]['耗时']} vs {recipe_db[loser]['耗时']}）"
        template = random.choice(COMPARE_TEMPLATES[mode])
        question = _safe_fill(template, a=a, b=b)
        cases.append(EvalCase(
            id=f"rc_{made:03d}",
            scenario="recipe_compare",
            turns=[question],
            expected_route="graphrag-query",
            expected_tool="cypher_query",
            expected_args={"dish_a": a, "dish_b": b, "compare": mode},
            expected_slots={"dish_a": a, "dish_b": b},
            relevant_ids=[a, b],
            expected_answer_keywords=[winner],
            ground_truth=gt,
            gt_source=f"recipe.json:{a}.{recipe_db[a]['口味']} vs recipe.json:{b}.{recipe_db[b]['口味']}",
            note=mode,
        ))
        made += 1
    return cases


def gen_stat_query(recipe_db: Dict[str, dict], n: int = 10) -> List[EvalCase]:
    """统计查询（Text2SQL）。ground truth = 生成时对知识库的真实计数。"""
    # 用"类型"字段的 Top 类目统计
    from collections import Counter
    counter = Counter(
        v["类型"] for v in recipe_db.values()
        if isinstance(v, dict) and v.get("类型")
    )
    cases = []
    keys = [k for k, _ in counter.most_common(30)]
    random.shuffle(keys)
    for i, key in enumerate(keys[:n]):
        count = counter[key]
        template = random.choice(STAT_TEMPLATES)
        question = _safe_fill(template, key=key)
        cases.append(EvalCase(
            id=f"sq_{i:03d}",
            scenario="stat_query",
            turns=[question],
            expected_route="text2sql-query",
            expected_tool="text2sql_query",
            expected_args={"group": key, "metric": "count"},
            expected_slots={"category": key},
            relevant_ids=[],
            expected_answer_keywords=[str(count)],
            ground_truth=str(count),
            gt_source=f"recipe.json:{key} 类目计数={count}",
            note=key,
        ))
    return cases


def gen_history_faq(
    recipe_db: Dict[str, dict],
    kb_text: str,
    n: int = 20,
    use_llm: bool = True,
) -> List[EvalCase]:
    """历史文化典故问答（kb-query / FAQ 类）。"""
    paras = split_kb_paragraphs(kb_text)
    cases = []
    if use_llm:
        qa_pairs = _llm_generate_faq(paras, n)
    else:
        # 无 LLM：段落首句截断做问题占位，人工后续补充
        qa_pairs = [(p[:30] + "是什么？", p[:80]) for p in paras[:n]]
    for i, (question, answer) in enumerate(qa_pairs):
        para = paras[i % len(paras)]
        topic = para.split("。")[0][:30]
        cases.append(EvalCase(
            id=f"hf_{i:03d}",
            scenario="history_faq",
            turns=[question],
            expected_route="kb-query",
            expected_tool="postgres",
            expected_args={"query": question},
            expected_slots={},
            relevant_ids=[f"kb_para_{i % len(paras)}"],
            topic=topic,
            expected_answer_keywords=[answer[:20]],
            ground_truth=answer,
            gt_source=f"data/kb/data.txt 段落 #{i % len(paras)}",
        ))
    return cases


def _llm_generate_faq(paras: List[str], n: int) -> List[Tuple[str, str]]:
    """用真实 LLM（配置的 API）从段落生成 自然问题 + 段落内答案。"""
    import os
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE") or os.environ.get("LLM_BASE_URL"),
    )
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    picked = paras[:n]
    results = []
    for p in picked:
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.3,
                messages=[
                    {"role": "system", "content":
                     "你是中文饮食文化知识出题器。根据用户提供的段落，生成 1 个用户会自然提出的问题"
                     "（简体中文，口语化，像普通食客问的），并要求回答内容完全来自该段落。"
                     '只输出 JSON：{"question": "...", "answer": "..."}'},
                    {"role": "user", "content": p},
                ],
                response_format={"type": "json_object"},
            )
            obj = json.loads(resp.choices[0].message.content)
            results.append((obj["question"], obj["answer"]))
        except Exception as e:  # noqa
            results.append((p[:30] + "是什么？", p[:80]))
    return results


# ---------------------------------------------------------------------------
# 手工场景（多轮记忆 + 负样本）
# ---------------------------------------------------------------------------

MANUAL_MULTI_TURN = [
    # (turns, expected_route, slots/note)
    (["宫保鸡丁怎么做？", "那鱼香肉丝呢？", "我刚才问的第一道菜是哪个？"],
     "graphrag-query", "多轮记忆：跨轮引用第一问"),
    (["麻婆豆腐需要哪些食材？", "这道菜辣不辣？", "把前面问过的食材再说一遍"],
     "graphrag-query", "多轮记忆：引用食材轮"),
    (["番茄炒蛋的做法？", "要放糖吗？", "鸡蛋要提前炒吗？"],
     "graphrag-query", "多轮记忆：同一主题追问"),
    (["红烧肉怎么做？", "糖色怎么炒？", "记住我说我要做给老人吃", "那少放点糖的做法是什么？"],
     "graphrag-query", "多轮记忆：偏好注入"),
    (["川菜有多少道菜？", "其中麻辣的有几道？", "我第一个问题问的是什么？"],
     "text2sql-query", "多轮记忆：统计引用"),
    (["宫保鸡丁的典故是什么？", "那它的做法呢？", "典故里提到哪个朝代？"],
     "kb-query", "多轮记忆：kb 交叉"),
    (["给我推荐一道下饭菜", "要辣的", "你刚才推荐的叫什么？"],
     "additional-query", "多轮记忆：追问后引用"),
    (["饺子怎么做？", "馅料怎么调？", "和面用冷水还是热水？", "把刚才和面的要点总结一下"],
     "graphrag-query", "多轮记忆：长对话总结"),
    (["鱼香肉丝是什么菜系？", "它的主料是什么？", "那鱼香是什么味道？"],
     "graphrag-query", "多轮记忆：属性连续追问"),
    (["清蒸鲈鱼需要什么材料？", "蒸多久？", "这两个问题分别问的什么？"],
     "graphrag-query", "多轮记忆：双问引用"),
]

MANUAL_NEGATIVE = [
    ("今天天气怎么样？", "general-query", "越界：天气"),
    ("帮我写一首关于春天的诗", "general-query", "越界：诗歌创作"),
    ("现在买什么股票好？", "general-query", "越界：金融"),
    ("推荐一部好看的电影", "general-query", "越界：电影"),
    ("讲个笑话", "general-query", "越界：笑话"),
    ("你叫什么名字？", "general-query", "闲聊"),
    ("1+1等于几？", "general-query", "越界：数学"),
    ("北京明天限行吗？", "general-query", "越界：交通"),
    ("帮我翻译这句话到英语", "general-query", "越界：翻译"),
    ("推荐个菜", "additional-query", "模糊：需要追问"),
    ("我想做菜", "additional-query", "模糊：需要追问"),
    ("有什么好吃的？", "additional-query", "模糊：需要追问"),
    ("帮我看看这个菜谱", "additional-query", "模糊：缺少输入"),
    ("怎么做？", "additional-query", "模糊：缺少对象"),
    ("推荐一个简单的", "additional-query", "模糊：需要追问"),
]


def gen_multi_turn() -> List[EvalCase]:
    cases = []
    for i, (turns, route, note) in enumerate(MANUAL_MULTI_TURN):
        cases.append(EvalCase(
            id=f"mt_{i:03d}",
            scenario="multi_turn",
            turns=turns,
            expected_route=route,
            expected_tool="none",
            note=note,
        ))
    return cases


def gen_negative() -> List[EvalCase]:
    cases = []
    for i, (q, route, note) in enumerate(MANUAL_NEGATIVE):
        cases.append(EvalCase(
            id=f"ng_{i:03d}",
            scenario="negative",
            turns=[q],
            expected_route=route,
            expected_tool="none",
            negative=True,
            note=note,
        ))
    return cases


# ---------------------------------------------------------------------------
# 抽样辅助
# ---------------------------------------------------------------------------

def _pick_with_coverage(recipe_db: Dict[str, dict], n: int, key: str = "类型") -> List[str]:
    """按 key 字段分层抽样，保证覆盖不同类目；过滤不适合出题的菜名。"""
    import re as _re
    bad = _re.compile(r"[#【】{}~()（）\U0001F300-\U0001FAFF\u2600-\u27BF]")
    names = [k for k in recipe_db
             if len(k) <= 20 and not bad.search(k) and not k.startswith("_")]
    random.shuffle(names)
    with_key = [k for k in names if isinstance(recipe_db[k], dict) and recipe_db[k].get(key)]
    without = [k for k in names if k not in with_key]
    pool = with_key + without
    return pool[:n]


# ---------------------------------------------------------------------------
# 难样本生成（同义改写 / 描述性需求 / 领域边界 / 复杂统计）
# 用于扩展评测集，避免"query 由知识库条目生成"导致的指标虚高
# ---------------------------------------------------------------------------

# 描述性菜谱查找（无完整菜名，需要图谱检索）
HARD_DESCRIPTIVE = [
    ("想做一道用鸡肉做的下饭菜，要辣一点的", "graphrag-query", "描述性查找：鸡肉+辣"),
    ("家里有豆腐和肉末，能做点什么菜？", "graphrag-query", "描述性查找：食材组合"),
    ("我想学一道宴客的大菜，要有面子", "graphrag-query", "描述性查找：场景"),
    ("给我推荐一道适合夏天的凉菜", "graphrag-query", "描述性查找：季节+类型"),
    ("有没有用电饭煲就能做的菜？", "graphrag-query", "描述性查找：工具"),
    ("孩子不爱吃蔬菜，有什么菜能把蔬菜藏起来？", "graphrag-query", "描述性查找：场景需求"),
    ("想给老人做道软烂好消化的菜", "graphrag-query", "描述性查找：人群"),
    ("冰箱里只有番茄和鸡蛋，能做几道菜？", "graphrag-query", "描述性查找：食材穷举"),
    ("减肥期间吃什么菜好？", "graphrag-query", "描述性查找：营养需求"),
    ("周末想烤个东西，有什么简单的？", "graphrag-query", "描述性查找：烤箱"),
]

# 同义改写（菜名不完整出现 / 口语化）
HARD_REFORMULATED = [
    ("这道经典的川味鸡丁菜，花生米应该什么时候下锅？", "graphrag-query", "同义改写：宫保鸡丁"),
    ("麻婆那款豆腐，肉末要提前腌吗？", "graphrag-query", "同义改写：麻婆豆腐"),
    ("糖醋那个排骨，是先炸还是先煮？", "graphrag-query", "同义改写：糖醋排骨"),
    ("那道红烧的肉，用五花还是后腿？", "graphrag-query", "同义改写：红烧肉"),
    ("水煮的那道鱼，最后淋油那步怎么操作？", "graphrag-query", "同义改写：水煮鱼"),
    ("大盘鸡里面的面要提前煮吗？", "graphrag-query", "同义改写：大盘鸡"),
    ("蚂蚁上树这道菜，粉丝要泡多久？", "graphrag-query", "同义改写：蚂蚁上树"),
    ("酸辣土豆丝怎么才能脆？", "graphrag-query", "同义改写：技巧"),
    ("西红柿炒鸡蛋，是先炒蛋还是先炒番茄？", "graphrag-query", "同义改写：顺序"),
    ("蛋炒饭怎么才能粒粒分明？", "graphrag-query", "同义改写：技巧"),
]

# 领域边界（含菜谱词但越界/需要追问——易误判）
HARD_BOUNDARY = [
    ("帮我写一个菜谱的模板，要有格式", "additional-query", "边界：文档模板"),
    ("辣条怎么做？", "general-query", "边界：辣条非菜谱"),
    ("奶茶算是菜吗？", "general-query", "边界：定义讨论"),
    ("给我推荐一家好吃的川菜馆", "general-query", "边界：线下餐馆"),
    ("减肥餐的卡路里怎么算？", "general-query", "边界：通用营养"),
    ("菜谱上的盐适量是几克？", "general-query", "边界：通用问题"),
    ("帮我列个一周的菜谱计划", "additional-query", "边界：计划生成需细节"),
    ("一道菜放多少油合适？", "general-query", "边界：通用烹饪问题"),
    ("外卖点什么菜健康？", "general-query", "边界：外卖"),
    ("菜能不能隔夜吃？", "general-query", "边界：食品安全常识"),
]

# 复杂统计（排名/最值——易误判 graphrag）
HARD_STAT = [
    ("做起来最快的一道菜是什么？", "text2sql-query", "统计：最值"),
    ("耗时最长的菜是哪道？", "text2sql-query", "统计：最值"),
    ("什么类型的菜最多？", "text2sql-query", "统计：众数"),
    ("辣口的菜多不多？", "text2sql-query", "统计：分组计数"),
    ("烘焙类的菜谱大概有几道？", "text2sql-query", "统计：类目计数"),
    ("十分钟就能搞定的菜有多少道？", "text2sql-query", "统计：条件计数"),
    ("哪个菜系收录的菜最多？", "text2sql-query", "统计：分组最值"),
    ("汤羹和凉菜哪个种类更多？", "text2sql-query", "统计：两组比较"),
    ("做菜超过一小时的菜谱占多少？", "text2sql-query", "统计：占比"),
    ("一共收录了多少种食材？", "text2sql-query", "统计：总计数"),
]

# 混淆（对比+属性混合，需要图谱推理）
HARD_CONFUSION = [
    ("宫保鸡丁和鱼香肉丝哪个是川菜？", "graphrag-query", "混淆：菜系归属对比"),
    ("红烧肉和回锅肉都需要炒糖色吗？", "graphrag-query", "混淆：工艺对比"),
    ("东坡肉和红烧肉有什么区别？", "graphrag-query", "混淆：近似菜对比"),
    ("饺子皮和馄饨皮做法一样吗？", "graphrag-query", "混淆：面食对比"),
    ("糖醋里脊和锅包肉哪个更脆？", "graphrag-query", "混淆：口感对比"),
    ("清蒸和红烧哪种做法更健康？", "graphrag-query", "混淆：做法比较"),
    ("麻婆豆腐和家常豆腐有什么不同？", "graphrag-query", "混淆：近似菜对比"),
    ("龙井虾仁和碧螺虾仁用的茶一样吗？", "graphrag-query", "混淆：食材对比"),
    ("白切鸡和口水鸡哪个是凉菜？", "graphrag-query", "混淆：冷热属性"),
    ("酸菜鱼和水煮鱼哪个更辣？", "graphrag-query", "混淆：辣度对比"),
]


def gen_hard_cases() -> List[EvalCase]:
    """难样本：描述性 / 同义改写 / 领域边界 / 复杂统计 / 混淆。"""
    cases = []

    def _add(seq, scenario, note_prefix, tool="none", gt=""):
        for i, (q, exp_route, note) in enumerate(seq):
            cases.append(EvalCase(
                id=f"{note_prefix}_{i:03d}",
                scenario=scenario,
                turns=[q],
                expected_route=exp_route,
                expected_tool=tool,
                expected_slots={},
                relevant_ids=[],
                expected_answer_keywords=[],
                ground_truth=gt,
                note=note,
            ))

    _add(HARD_DESCRIPTIVE, "recipe_search", "hd")
    _add(HARD_REFORMULATED, "recipe_search", "hr")
    _add(HARD_BOUNDARY, "negative", "hb")
    _add(HARD_STAT, "stat_query", "hs")
    _add(HARD_CONFUSION, "recipe_compare", "hc")
    return cases


# ---------------------------------------------------------------------------
# 总入口
# ---------------------------------------------------------------------------

def generate_all(
    count: Optional[int] = None,
    use_llm: bool = False,
    seed: int = 42,
    include_hard: bool = True,
) -> List[EvalCase]:
    global random
    random.seed(seed)

    recipe_db = load_recipe_db()
    kb_text = load_kb_text()

    cases: List[EvalCase] = []
    cases += gen_recipe_search(recipe_db, n=30)
    cases += gen_recipe_detail(recipe_db, n=25)
    cases += gen_recipe_compare(recipe_db, n=15)
    cases += gen_stat_query(recipe_db, n=10)
    cases += gen_history_faq(recipe_db, kb_text, n=20, use_llm=use_llm)
    cases += gen_multi_turn()
    cases += gen_negative()
    if include_hard:
        cases += gen_hard_cases()

    if count is not None:
        cases = cases[:count]
    return cases
