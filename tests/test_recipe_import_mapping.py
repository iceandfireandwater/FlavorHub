"""
Offline mapping tests for recipe batch import tooling.

These tests intentionally avoid external services (Milvus/LLM/network).
They only validate deterministic transforms from source data -> recipe payloads
that the HTTP API accepts.
"""


import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_scripts_import_recipes_cli_exists() -> None:
    assert (PROJECT_ROOT / "scripts" / "import_recipes.py").exists()


def test_recipe_json_entry_to_recipe_basic_mapping() -> None:
    from gustobot.infrastructure.knowledge.recipe_import import recipe_json_entry_to_recipe

    entry = {
        "主食材": [["香肠", "2根"], ["菜干", "200g"]],
        "辅料": [["豆豉", "2匙"], ["蒜", "少许"]],
        "耗时": "十分钟",
        "口味": "酱香",
        "工艺": "炒",
        "做法": "1:准备的食材。2:香肉肠切片。3:爆香蒜末、豆豉。4:成品。",
        "类型": "家常菜",
    }

    recipe = recipe_json_entry_to_recipe("香肠炒菜干", entry)

    assert recipe["name"] == "香肠炒菜干"
    assert recipe["category"] == "家常菜"
    assert recipe["time"] == "十分钟"
    assert "香肠 2根" in recipe["ingredients"]
    assert "菜干 200g" in recipe["ingredients"]
    assert "豆豉 2匙" in recipe["ingredients"]
    assert recipe["steps"][:2] == ["准备的食材。", "香肉肠切片。"]
    assert "口味" in (recipe.get("tips") or "")
    assert "工艺" in (recipe.get("tips") or "")


def test_wikipedia_page_to_recipe_mapping() -> None:
    from gustobot.crawler.wikipedia import wikipedia_page_to_recipe

    page = {
        "title": "川菜",
        "extract": "川菜是中国八大菜系之一，以麻辣著称。",
        "fullurl": "https://zh.wikipedia.org/wiki/%E5%B7%9D%E8%8F%9C",
    }

    recipe = wikipedia_page_to_recipe(page, query="川菜")

    assert recipe["name"] == "川菜"
    assert recipe["category"].startswith("Wikipedia")
    assert recipe["steps"] == ["川菜是中国八大菜系之一，以麻辣著称。"]
    assert "wikipedia.org" in (recipe.get("tips") or "")
