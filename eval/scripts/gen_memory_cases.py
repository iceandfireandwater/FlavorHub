# -*- coding: utf-8 -*-
"""生成记忆评测集（v2，全面重造）——可重复执行，结果确定。

产出：
1. eval/cases/memory_eval.jsonl       跨会话记忆，50 条
2. eval/cases/intra_memory_eval.jsonl 会话内多轮记忆，50 条

v2 相比 v1 的三项改进：
A. 覆盖 6 类原先缺失的场景：记忆更新 / 冲突 / 负例 / 长程 / 多项并存 / 隔离性
B. 判据强化：expect_all（必须全命中）+ expect_any（至少一个，且普遍给多个关键词）
   + forbid_any（防"问句蒙分"）
C. 语料扩充：fillers 30+ 种、开头措辞 10+ 种，避免模板同质化

用法：python -m eval.gen_memory_cases
"""
from __future__ import annotations

import json
from pathlib import Path

CASES_DIR = Path(__file__).resolve().parent / "cases"

# ── 素材池 ────────────────────────────────────────────────────
# (姓名, 城市, 职业, 多种自述措辞)
PEOPLE = [
    ("阿强", "成都", "程序员", ["我叫阿强，住在成都，是个程序员", "我是阿强，在成都写代码", "大家都叫我阿强，成都人，做开发的"]),
    ("小雨", "上海", "设计师", ["我叫小雨，在上海做设计", "我是小雨啦，人在上海，职业是设计师"]),
    ("老王", "西安", "退休教师", ["我叫老王，在西安，退休前是中学老师", "我是老王，西安人，退休教师"]),
    ("陈叔", "长沙", "出租车司机", ["我是陈叔，跑长沙的出租", "大家都叫我陈叔，在长沙开出租车"]),
    ("李阿姨", "杭州", "家庭主妇", ["大家叫我李阿姨，我住杭州", "我是李阿姨，杭州人，在家带孩子做饭"]),
    ("小林", "深圳", "大学生", ["我叫小林，在深圳读大学", "我是小林，深圳的大学生"]),
    ("老周", "重庆", "厨师", ["我姓周，在重庆做厨师十几年了", "我是老周，重庆的厨师，干了十几年"]),
    ("米粒", "南京", "护士", ["我网名叫米粒，在南京当护士", "我是米粒，人在南京，职业是护士"]),
    ("阿豪", "厦门", "自由职业", ["我是阿豪，在厦门做自由职业", "我叫阿豪，厦门，自由职业者"]),
    ("甜甜", "青岛", "会计", ["我叫甜甜，在青岛做会计", "我是甜甜，青岛的一名会计"]),
    ("老赵", "哈尔滨", "货车司机", ["我是老赵，哈尔滨的，跑长途货运", "我姓赵，哈尔滨人，开长途货车"]),
    ("桃子", "昆明", "小学老师", ["我叫桃子，在昆明教书", "我是桃子，昆明的小学老师"]),
    ("大山", "拉萨", "导游", ["我叫大山，在拉萨做导游", "我是大山，拉萨的导游，带了八年团"]),
    ("小美", "苏州", "咖啡师", ["我是小美，在苏州开咖啡店", "我叫小美，苏州有一家自己的咖啡店"]),
    ("阿峰", "天津", "机械工程师", ["我叫阿峰，在天津做机械工程师", "我是阿峰，天津，机械工程师"]),
    ("大牛", "郑州", "外卖骑手", ["我叫大牛，在郑州送外卖", "我是大牛，郑州的外卖小哥"]),
]

# (约束A, 约束B, 自述)
UPDATES = [
    # (要解除的东西, 原始表述, 追问句) —— 追问强制二元回答，
    # 避免 "不能吃" 里含 "能吃" 造成假命中。
    # 注意：素材必须是**可撤回的偏好**，不能用身体状况/过敏（"乳糖不耐""对鸡蛋过敏"
    # 不会因为用户改主意而消失，模型答"不可以"反而是对的，会造成假失败）。
    ("香菜", "我不吃香菜", "我现在可以吃香菜了吗？请只回答「可以」或「不可以」。"),
    ("辣", "我最近不吃辣，想清淡点", "我现在可以吃辣了吗？请只回答「可以」或「不可以」。"),
    ("动物内脏", "我不吃动物内脏", "我现在可以吃动物内脏了吗？请只回答「可以」或「不可以」。"),
    ("鸡蛋", "我最近不太想吃鸡蛋", "我现在可以吃鸡蛋了吗？请只回答「可以」或「不可以」。"),
    ("牛奶", "我最近不喝牛奶", "我现在可以喝牛奶了吗？请只回答「可以」或「不可以」。"),
    ("花生", "我最近在避开花生", "我现在可以吃花生了吗？请只回答「可以」或「不可以」。"),
    ("海鲜", "我最近不吃海鲜", "我现在可以吃海鲜了吗？请只回答「可以」或「不可以」。"),
    ("生冷", "我最近不吃生冷的东西", "我现在可以吃生冷的东西了吗？请只回答「可以」或「不可以」。"),
    ("豆制品", "我最近不吃豆制品", "我现在可以吃豆制品了吗？请只回答「可以」或「不可以」。"),
    ("葱姜蒜", "我做菜不放葱姜蒜", "我现在可以放葱姜蒜了吗？请只回答「可以」或「不可以」。"),
]

CONSTRAINTS = [
    ("花生", "不吃辣", "我不吃辣，另外对花生过敏"),
    ("素食", "葱姜蒜", "我吃素，而且不吃葱姜蒜"),
    ("高血压", "太咸", "我有高血压，做菜不能太咸"),
    ("糖尿病", "糖", "家里有糖尿病人，做菜不能放糖"),
    ("孕妇", "生冷", "我怀孕了，不吃生冷的东西"),
    ("乳糖不耐", "牛奶", "我乳糖不耐，不能喝牛奶"),
    ("痛风", "海鲜", "我有痛风，不能吃海鲜"),
    ("胃病", "油腻", "我胃不好，吃不了太油腻的"),
    ("鸡蛋过敏", "鸡蛋", "孩子对鸡蛋过敏，不能放鸡蛋"),
    ("香菜", "动物内脏", "我不吃香菜，也不吃动物内脏"),
]
CONSTRAINT_LINES = [
    "我不吃辣，另外对花生过敏", "我吃素，而且不吃葱姜蒜", "我有高血压，做菜不能太咸",
    "家里有糖尿病人，做菜不能放糖", "我怀孕了，不吃生冷的东西", "我乳糖不耐，不能喝牛奶",
    "我有痛风，不能吃海鲜", "我胃不好，吃不了太油腻的", "孩子对鸡蛋过敏，不能放鸡蛋",
    "我不吃香菜，也不吃动物内脏", "医生让我低盐低脂", "我吃清真，不吃猪肉",
    "我不吃牛肉和羊肉", "我刚做完手术，得吃清淡的", "我对虾蟹过敏",
]

PREFERENCES = [
    ("清淡", "少油", "我口味清淡，喜欢少油少盐"),
    ("酸甜", "番茄", "我偏爱酸甜口，特别爱吃番茄"),
    ("面食", "馒头", "我特别爱吃面食，经常自己蒸馒头"),
    ("煲汤", "老火汤", "我喜欢煲汤，经常炖老火汤"),
    ("重口", "辣", "我口味重，无辣不欢"),
    ("蒸菜", "油炸", "我喜欢蒸菜，很少做油炸的"),
    ("烘焙", "蛋糕", "我爱烘焙，常做蛋糕和面包"),
    ("炖菜", "慢炖", "我喜欢炖菜，爱用小火慢炖"),
    ("凉拌", "沙拉", "我爱吃凉拌菜和沙拉"),
    ("粥品", "小米粥", "我早餐喜欢喝粥，常熬小米粥"),
    ("海鲜", "清蒸", "我爱吃海鲜，喜欢清蒸"),
    ("烧烤", "孜然", "我喜欢烧烤味，爱放孜然"),
    ("川菜", "花椒", "我偏爱川菜，喜欢花椒的麻"),
    ("日料", "生食", "我喜欢日料，爱吃生鲜"),
]

DISHES = [
    ("红烧肉", "五花肉"), ("西红柿炒鸡蛋", "西红柿"), ("宫保鸡丁", "鸡胸肉"),
    ("清蒸鲈鱼", "鲈鱼"), ("麻婆豆腐", "豆腐"), ("糖醋里脊", "里脊"),
    ("冬瓜排骨汤", "排骨"), ("青椒肉丝", "青椒"), ("蒜蓉西兰花", "西兰花"),
    ("可乐鸡翅", "鸡翅"), ("酸辣土豆丝", "土豆"), ("香菇滑鸡", "香菇"),
]

# 扩充后的 filler 语料池（30+ 种，避免同一条里重复出现）
FILLERS = [
    "推荐一道菜吧", "今天想喝点汤", "随便问一句，谢谢", "你平时都做什么菜",
    "来点凉菜吧", "有没有适合早餐的", "再说两个家常菜", "还有什么别的推荐",
    "中午吃什么好", "想学个简单的", "有没有下饭的", "周末想做点好的",
    "适合带饭的有吗", "做给两个人吃", "想少放点油的", "有没有半小时能做完的",
    "想试试蒸的", "有没有不用烤箱的", "再来一道甜的", "想学个汤的做法",
    "有什么快手菜", "想给朋友做一桌", "减脂能吃的有吗", "孩子爱吃的有吗",
    "有没有适合拌面的", "想学个卤味", "家里只有鸡蛋怎么办", "想学凉拌菜",
    "有没有省事的做法", "想换个口味", "有汤有菜推荐一下", "想学个面点",
]

# reference 类专用：追问"我刚才说的那道菜是什么"时，filler 里若含**烹饪方式或菜品名**
# （"想试试蒸的""想学凉拌菜""想学个汤的做法"）就会变成竞争答案 —— 实测
# intra_reference_10 把"可乐鸡翅"答成了"蒸的做法"。这里过滤掉这类词。
_AMBIGUOUS = ("蒸", "炖", "凉拌", "卤", "面点", "汤的做法", "甜的", "炒", "烤", "快手菜")
NEUTRAL_FILLERS = [f for f in FILLERS if not any(w in f for w in _AMBIGUOUS)]



# 判据别名：素材表的"标签词"与"用户原话里的说法"经常不是同一个词
# （素食 vs 吃素、孕妇 vs 怀孕、胃病 vs 胃不好、老周 vs 我姓周），
# 而判据做的是字面子串匹配 → 模型答对了却判失败（实测 8 条失败里 5 条是这种）。
# 这里把标签展开成"可接受的多种表述"，生成用例时写进 expect_any/expect_all。
ALIASES = {
    # —— 约束类 ——
    "素食": ["吃素", "素食", "素的", "不吃肉"],
    "孕妇": ["怀孕", "孕妇", "孕期"],
    "胃病": ["胃不好", "胃病", "肠胃"],
    "太咸": ["太咸", "咸", "少盐", "少放盐", "少放点盐", "清淡", "控盐"],
    "生冷": ["生冷", "生食", "凉的", "凉拌", "冰"],
    "高血压": ["高血压", "血压"],
    "糖尿病": ["糖尿病", "血糖"],
    "乳糖不耐": ["乳糖", "不耐"],
    "痛风": ["痛风", "嘌呤"],
    "鸡蛋过敏": ["鸡蛋", "过敏"],
    "花生": ["花生", "坚果"],
    "不吃辣": ["辣"],
    "葱姜蒜": ["葱姜蒜", "葱姜", "蒜"],
    "油腻": ["油腻", "少油", "少放油", "油"],
    "香菜": ["香菜"],
    "动物内脏": ["内脏"],
    "牛奶": ["牛奶", "奶"],
    "海鲜": ["海鲜"],
    "糖": ["糖"],
    # —— 人设类：老周的"老周"只存在于标签，用户原话是"我姓周" ——
    "老周": ["周"],
    "老赵": ["赵"],
}


def expand(terms):
    """expect_any 用：把标签展开成**扁平的同义词并集**，命中任意一个即算通过。"""
    out = []
    for t in terms or []:
        out.append(t)
        for alias in ALIASES.get(t, []):
            if alias != t and alias not in out:
                out.append(alias)
    return out


def expand_groups(terms, extra_terms=None):
    """expect_all 用：每个标签展开成**一个同义词组**，判定是"每组命中其一"。

    为什么不能像 expect_any 那样压平：expect_all 的语义是"必须全部命中"，
    如果同义词也压进同一个扁平列表，就等于要求模型把「痛风/嘌呤」「海鲜/海产」
    每个说一遍——永远不可能满足，导致 multi 类 7/10 条被误判为失败。
    """
    groups = [[t] + [a for a in ALIASES.get(t, []) if a != t] for t in (terms or [])]
    for t in (extra_terms or []):
        groups.append([t] + [a for a in ALIASES.get(t, []) if a != t])
    return groups


def _pick(pool, i):
    return pool[i % len(pool)]


def build_cross_cases() -> list[dict]:
    """跨会话：会话 A 自述 → 另开会话 B 追问。80 条。

    类型分布：facts 16 / constraint 14 / preference 10 / no_poison 10 /
              update 10 / multi 10 / isolation 10
    """
    cases: list[dict] = []

    def add(kind, **kw):
        if kw.get("expect_any"):
            kw["expect_any"] = expand(kw["expect_any"])
        if kw.get("expect_all"):
            kw["expect_all"] = expand_groups(kw["expect_all"])
        cases.append({"id": f"mem_{kind}_{sum(1 for c in cases if c['kind'] == kind) + 1:02d}",
                      "kind": kind, **kw})

    # 1) facts 16：一半只问城市（宽松），一半要求姓名+城市全中（严格）
    for i in range(16):
        name, city, job, lines = _pick(PEOPLE, i)
        if i % 2 == 0:
            add("facts", turns=[_pick(lines, i)], probe=f"{name}来自哪里？",
                expect_any=[city], forbid_any=[])
        else:
            add("facts", turns=[_pick(lines, i)],
                probe="你还记得我的名字和所在城市吗？",
                expect_all=[name, city], forbid_any=[])

    # 2) constraint 14
    for i in range(14):
        a, b, line = _pick(CONSTRAINTS, i)
        add("constraint", turns=[line, _pick(FILLERS, i)],
            probe="我之前说过什么忌口或饮食要求？", expect_any=[a, b])

    # 3) preference 10
    for i in range(10):
        a, b, line = _pick(PREFERENCES, i)
        add("preference", turns=[line, _pick(FILLERS, i + 3)],
            probe="我的口味偏好是什么？", expect_any=[a, b])

    # 4) no_poison 10：不能把"没提到"固化成结论
    for i in range(10):
        name, city, job, lines = _pick(PEOPLE, i)
        add("no_poison", turns=[_pick(lines, i + 1)],
            probe="我之前有没有提过我自己的名字？", expect_any=[name],
            forbid_any=["没有提到", "没有提过", "并没有提到", "无法提供", "没有说过"])

    # 5) update 10：用户明确改主意后，旧限制必须被解除
    #   判据用强制二元回答：expect「可以」/ forbid「不可以」。
    #   旧写法有三个坑：probe 里出现"忌口"、forbid 里放裸"不能"、expect 里放"能吃"
    #   （"不能吃" 含 "能吃" → 假命中），三处叠加导致 10 条全错。
    for i in range(10):
        item, orig, ask = _pick(UPDATES, i)
        add("update",
            turns=[orig, f"我改主意了，{item}那个限制现在不用管了"],
            probe=ask, expect_any=["可以"], forbid_any=["不可以"])

    # 6) multi 10：一次说多个约束，追问时必须全部记住
    for i in range(10):
        a, b, line = _pick(CONSTRAINTS, i + 3)
        add("multi", turns=[line, _pick(FILLERS, i + 8)],
            probe="把我说过的要求都列一下", expect_all=[a, b])

    # 7) isolation 10：**别的用户**说过的内容，本用户不该知道
    for i in range(10):
        name, city, job, lines = _pick(PEOPLE, i + 6)
        add("isolation", other_user_lines=[_pick(lines, i)], turns=["我不吃辣"],
            probe=f"你知道有个人叫{name}吗？他是做什么的？", forbid_any=[city, job])
    return cases[:80]


def _fillers(start: int, n: int) -> list[str]:
    """取 n 条 filler（循环取用，天然去重到池子大小）。"""
    return [FILLERS[(start + k) % len(FILLERS)] for k in range(n)]


def build_intra_cases() -> list[dict]:
    """会话内多轮：全程同一会话，前面说的话滑出窗口(5 轮)后还能否被记住。80 条。

    类型分布：intra_constraint 16 / intra_facts 16 / intra_reference 12 /
              intra_preference 10 / longrange 10 / conflict 8 / negative 8
    """
    cases: list[dict] = []

    def add(kind, **kw):
        if kw.get("expect_any"):
            kw["expect_any"] = expand(kw["expect_any"])
        if kw.get("expect_all"):
            kw["expect_all"] = expand_groups(kw["expect_all"])
        cases.append({"id": f"intra_{kind}_{sum(1 for c in cases if c['kind'] == kind) + 1:02d}",
                      "kind": kind, **kw})

    # 1) 约束类 16：说一次约束 → 5 轮 filler → 追问
    for i in range(16):
        a, b, line = _pick(CONSTRAINTS, i)
        add("constraint", setup=[line], fillers=_fillers(i, 6),
            probe="我前面说的饮食要求，你还记得吗？", expect_any=[a, b])

    # 2) 身份类 16
    for i in range(16):
        name, city, job, lines = _pick(PEOPLE, i)
        add("facts", setup=[_pick(lines, i), f"我是做{job}的"], fillers=_fillers(i + 2, 6),
            probe="我叫什么？在哪里？做什么工作？", expect_all=[name, city])

    # 3) 多轮指代 12：先说菜名，隔几轮后用"刚才那道菜"追问
    for i in range(12):
        dish, main = _pick(DISHES, i)
        add("reference", setup=[f"我想做{dish}", f"{dish}主要用什么食材"],
            fillers=[NEUTRAL_FILLERS[(i + 4 + k) % len(NEUTRAL_FILLERS)] for k in range(6)],
            probe="我刚才说的那道菜是什么？", expect_any=[dish])

    # 4) 偏好类 10
    for i in range(10):
        a, b, line = _pick(PREFERENCES, i)
        add("preference", setup=[line], fillers=_fillers(i + 6, 6),
            probe="我的口味偏好是什么？", expect_any=[a, b])

    # 5) longrange 10（新场景）：**15 轮前**说的话，考验深层压缩
    for i in range(10):
        a, b, line = _pick(CONSTRAINTS, i + 2)
        add("longrange", setup=[line], fillers=_fillers(i, 15),
            probe="我最早跟你说的饮食要求是什么？", expect_any=[a, b])

    # 6) conflict 8（新场景）：同一会话内前后矛盾，应采信更正后的说法
    for i in range(8):
        a, b, line = _pick(CONSTRAINTS, i)
        add("conflict", setup=[line, f"不好意思刚才说错了，其实是{b}的问题，不是{a}"],
            fillers=_fillers(i + 7, 6),
            probe="我到底对什么过敏或忌口？", expect_any=[b],
            # 不用 forbid_any：模型"正确地指出某信息没提过"（如"关于过敏食材您没提到过"）
            # 会被"没有提到"误杀，而这与"把没提过的事当结论"语义相反，判据无法用
            # 字面匹配区分。expect_any 命中即说明它采纳了更正。
            forbid_any=[])

    # 7) negative 8（新场景）：闲聊内容不该被当成用户的偏好/要求
    for i in range(8):
        chit = _pick(["今天天气真不错", "谢谢你啦", "刚看完一场电影", "最近有点忙",
                      "路上堵车堵了好久", "周末打算出去走走", "刚睡醒", "心情不错"], i)
        a, b, line = _pick(CONSTRAINTS, i + 1)
        add("negative", setup=[chit, line], fillers=_fillers(i + 5, 6),
            probe="我有什么忌口或饮食要求？", expect_any=[a],
            forbid_any=["天气", "电影", "堵车", "心情"])
    return cases[:80]


def main() -> None:
    import random

    CASES_DIR.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(42)   # 固定种子：打乱但可复现
    for fname, cases in [("memory_eval.jsonl", build_cross_cases()),
                         ("intra_memory_eval.jsonl", build_intra_cases())]:
        rnd.shuffle(cases)          # 打散：避免同类型题扎堆
        with open(CASES_DIR / fname, "w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        from collections import Counter
        print("%-24s %3d 条  %s" % (fname, len(cases), dict(Counter(c["kind"] for c in cases))))


if __name__ == "__main__":
    main()
