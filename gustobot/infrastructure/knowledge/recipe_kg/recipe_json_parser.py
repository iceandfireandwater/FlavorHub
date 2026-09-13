"""
Shared data parsing utilities for recipe knowledge graph bootstrapping.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class IngredientAmount:
    name: str
    amount: Optional[str]
    role: str


@dataclass
class StepRecord:
    order: int
    instruction: str


@dataclass
class RecipeRecord:
    name: str
    cook_time: Optional[str]
    flavors: List[str]
    methods: List[str]
    dish_types: List[str]
    instructions: Optional[str]
    steps: List[StepRecord]
    main_ingredients: List[IngredientAmount]
    aux_ingredients: List[IngredientAmount]


@dataclass
class IngredientProfile:
    name: str
    nutrition: Optional[str]
    benefits: List[str]

# recipe.json
def load_recipe_records(recipe_json: Path) -> Tuple[List[RecipeRecord], Set[str]]:
    """Load structured recipe records from a JSON mapping."""
    payload = _load_json(recipe_json)
    return _normalise_recipes(payload)

# excipient.json
def load_ingredient_profiles(
    ingredient_json: Optional[Path],
    ingredients_used: Set[str],
) -> List[IngredientProfile]:
    """Return nutrition profiles for the subset of ingredients used in recipes."""
    if not ingredient_json or not ingredient_json.is_file():
        return []

    payload = _load_json(ingredient_json)
    profiles: List[IngredientProfile] = []
    for name in sorted(ingredients_used):
        raw = payload.get(name)
        if not isinstance(raw, dict):
            continue
        nutrition = _clean_text(raw.get("营养价值"))
        benefits = _split_benefits(_clean_text(raw.get("食用功效")))
        profiles.append(IngredientProfile(name=name, nutrition=nutrition, benefits=benefits))
    return profiles


# --------------------------------------------------------------------------- #
# Internal helpers shared by import/export pipelines
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON dataset not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level object to be a dict in {path}")
    return payload


def _normalise_recipes(
    recipes_raw: Dict[str, Dict[str, object]],
) -> Tuple[List[RecipeRecord], Set[str]]:
    records: List[RecipeRecord] = []
    ingredients_seen: Set[str] = set()

    for raw_name, payload in recipes_raw.items():
        if not isinstance(payload, dict):
            continue

        name = _clean_name(raw_name)
        cook_time = _clean_text(payload.get("耗时"))
        flavors = _split_multi(payload.get("口味"))
        methods = _split_multi(payload.get("工艺"))
        dish_types = _split_multi(payload.get("类型"))
        instructions = _clean_text(payload.get("做法"))

        main_ingredients = _normalise_ingredients(payload.get("主食材"), role="main", seen=ingredients_seen)
        aux_ingredients = _normalise_ingredients(payload.get("辅料"), role="aux", seen=ingredients_seen)
        steps = _normalise_steps(instructions)

        records.append(
            RecipeRecord(
                name=name,
                cook_time=cook_time,
                flavors=flavors,
                methods=methods,
                dish_types=dish_types,
                instructions=instructions,
                steps=steps,
                main_ingredients=main_ingredients,
                aux_ingredients=aux_ingredients,
            ),
        )

    return records, ingredients_seen


# " 红烧肉 "	"红烧肉"
# "宫保 鸡丁"	"宫保 鸡丁"（中间压缩成 1 空格）
def _clean_name(value: object) -> str:
    text = str(value or "").strip()  # 去首尾空白
    text = re.sub(r"\s+", " ", text)  # 中间连续空白 → 单个空格
    return text


# None	None
# " 十分钟 "	"十分钟"
# " "（纯空格）	None
def _clean_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()  # 去首尾空白
    return text or None

# "麻辣、香辣"	["麻辣", "香辣"]
# "炒,煎；炸"	["炒", "煎", "炸"]
# " " / None	[]
def _split_multi(value: object) -> List[str]:
    if not value:
        return []
    text = str(value)
    parts = re.split(r"[、/,，；;]+", text)  # 按顿号/斜杠/逗号/分号拆
    return [part.strip() for part in parts if part.strip()]  # 去首尾空白


def _normalise_ingredients(
    value: object,
    *,
    role: str,
    seen: Set[str],
) -> List[IngredientAmount]:
    result: List[IngredientAmount] = []
    items: Iterable[object]
    if isinstance(value, list):
        items = value
    else:
        items = []

    for item in items:
        # 有成分名，陈分量
        if isinstance(item, (list, tuple)) and item:
            name = _clean_name(item[0])  # 获取成分名
            amount = _clean_text(item[1]) if len(item) > 1 else None  # 获取数量
        # 只有成分名，没有成分量
        elif isinstance(item, str):
            name = _clean_name(item)  # 获取成分名
            amount = None   # 没有数量
        # 啥也没有
        else:
            continue
        if not name:
            continue
        seen.add(name)
        result.append(IngredientAmount(name=name, amount=amount, role=role))
    return result


def _normalise_steps(instructions: Optional[str]) -> List[StepRecord]:
    if not instructions:
        return []

    normalised = instructions.replace("：", ":")  # 替换中文冒号为英文冒号
    pattern = re.compile(r"(?P<order>\d+):\s*(?P<text>.*?)(?=(?:\d+:)|$)", re.S)
    matches = list(pattern.finditer(normalised))

    steps: List[StepRecord] = []
    
    # 有1:,2:这样的编号
    if matches:
        for match in matches:
            order = int(match.group("order"))  # 字符串"1" → 整数 1（图谱里 order 是 INTEGER）
            text = match.group("text").strip().rstrip("。.")  # 去首尾空白 + 去掉末尾句号
            if text:  # 空步骤丢弃（如"3:"后面没内容）
                steps.append(StepRecord(order=order, instruction=text))
    # 无编号
    else:
        fragments = re.split(r"[。.!？！\n]+", normalised)  # 按照句号、感叹号、问号、换行符拆分
        filtered = [fragment.strip() for fragment in fragments if fragment.strip()]
        for index, fragment in enumerate(filtered, start=1):
            steps.append(StepRecord(order=index, instruction=fragment))  # 再编号
    return steps


# "1、健脑：...\n2、预防过敏：...\n3、缓解失眠：..."
# → ['健脑：...', '预防过敏：...', '缓解失眠：...']
def _split_benefits(text: Optional[str]) -> List[str]:
    if not text:
        return []

    lines = re.split(r"[\n\r]+", text)  # 按照换行符拆分
    benefits: List[str] = []

    for line in lines:
        cleaned = line.strip()  # 去首尾空白

        # 去编号前缀
        # 原始行首	   匹配	 删后
        # 1、理气血...	1、	理气血...
        # 2. 抗真菌...	2.	抗真菌...
        # 3:缓解失眠...	3:	缓解失眠...
        # 4）健脑...	4）	健脑...
        cleaned = re.sub(r"^[0-9]+[\\.、:：)\s]*", "", cleaned)  # 去掉编号和句号
        if cleaned:
            benefits.append(cleaned)
    return benefits
