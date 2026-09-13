"""
Utility helpers to bootstrap the recipe knowledge graph from JSON sources.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

from .graph_database_client import Neo4jDatabase
from .recipe_json_parser import (
    IngredientProfile,
    RecipeRecord,
    load_ingredient_profiles,
    load_recipe_records,
)


class RecipeGraphImporter:
    """Load recipes and ingredient metadata from JSON files into Neo4j."""

    def __init__(self, database: Neo4jDatabase, batch_size: int = 200) -> None:
        self._database = database
        self._batch_size = max(50, batch_size)  # 批量下限 50，防止传 1 导致 2 万次请求

    def bootstrap_from_json(
        self,
        recipe_json: Path,
        ingredient_json: Optional[Path] = None,
        *,
        force: bool = False,
    ) -> bool:
        """Populate the graph from JSON sources."""
        try:
            recipes, ingredients_used = load_recipe_records(recipe_json)
        except FileNotFoundError:
            logger.warning(f"Recipe JSON not found at {recipe_json}, skipping import.")
            return False
        except Exception as exc:
            logger.error(f"Failed to parse recipe JSON: {exc}")
            return False

        if not recipes:
            logger.info("No recipe records found; skipping Neo4j bootstrap.")
            return False

        # Neo4j 有数据了就跳过
        if not force and not self._is_graph_empty():
            logger.info("Neo4j dataset already populated; skipping bootstrap.")
            return False

        # force = true，就清理图谱，强制刷新 Neo4j    
        if force:
            logger.info("Forcing recipe graph reload from JSON.")
            self._database.execute("MATCH (n) DETACH DELETE n")
        else:
            logger.info("Recipe graph is empty; importing dataset from JSON.")

        # 只取用到的食材
        profiles = load_ingredient_profiles(ingredient_json, ingredients_used)

        # 创建菜谱节点
        self._create_recipe_nodes(recipes)
        self._create_relationships(recipes)
        if profiles:
            self._attach_ingredient_metadata(profiles)

        logger.info(
            "Imported %s recipes and %s unique ingredients into Neo4j.",
            len(recipes),
            len(ingredients_used),
        )
        return True

    def _is_graph_empty(self) -> bool:
        query = "MATCH (n:Dish) RETURN COUNT(n) AS count"
        result = self._database.fetch(query)
        count = result[0]["count"] if result else 0
        return count == 0

    # 把 19,666 条菜谱切成 200 条一批，每批一次 Cypher 请求（UNWIND $batch 一次处理一批）——2 万条只用 ~100 次请求，而不是 2 万次
    def _chunked(self, data: Iterable[Dict[str, Any]]) -> Iterable[List[Dict[str, Any]]]:
        batch: List[Dict[str, Any]] = []
        for record in data:
            batch.append(record)
            if len(batch) >= self._batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    # 建 Dish/Flavor/CookingMethod/DishType + HAS_FLAVOR/USES_METHOD/BELONGS_TO_TYPE
    def _create_recipe_nodes(self, recipes: List[RecipeRecord]) -> None:
        query = """
        UNWIND $batch AS dish                        -- 把本批 200 道菜逐行展开，每行叫 dish
        MERGE (d:Dish {name: dish.name})             -- 按菜名找 Dish 节点，没有就建（幂等）
        SET d.cook_time = dish.cook_time,            -- 写入：耗时
            d.instructions = dish.instructions       --       做法全文
        WITH d, dish                                 -- 把 d 和 dish 传给下面（FOREACH 前必须有 WITH），d是Neo4j里的节点名，dish是传入的原始菜谱数据
        FOREACH (flavor IN dish.flavors |            -- 循环这道菜的所有口味（如 [麻辣, 咸鲜]）
            MERGE (f:Flavor {name: flavor})          --   口味节点：不存在才建
            MERGE (d)-[:HAS_FLAVOR]->(f)             --   建关系：菜 -有口味-> 口味节点
        )
        FOREACH (method IN dish.methods |            -- 同理：工艺
            MERGE (m:CookingMethod {name: method})
            MERGE (d)-[:USES_METHOD]->(m)
        )
        FOREACH (dtype IN dish.dish_types |          -- 同理：类型
            MERGE (t:DishType {name: dtype})
            MERGE (d)-[:BELONGS_TO_TYPE]->(t)
        )
        """

    #     (Dish: 麻婆豆腐) ──HAS_FLAVOR──▶ (Flavor: 麻辣)
    #    │            ──HAS_FLAVOR──▶ (Flavor: 咸鲜)
    #    │            ──USES_METHOD──▶ (CookingMethod: 烧)
    #    └─────────────BELONGS_TO_TYPE──▶ (DishType: 热菜)

        serialised = [
            {
                "name": record.name,
                "cook_time": record.cook_time,
                "instructions": record.instructions,
                "flavors": record.flavors,
                "methods": record.methods,
                "dish_types": record.dish_types,
            }
            for record in recipes
        ]

        for batch in self._chunked(serialised):
            self._database.execute(query, {"batch": batch})

    # 建 CookingStep/Ingredient + HAS_STEP/HAS_MAIN_INGREDIENT/HAS_AUX_INGREDIENT
    def _create_relationships(self, recipes: List[RecipeRecord]) -> None:
        step_query = """
        UNWIND $batch AS dish
        MATCH (d:Dish {name: dish.name})             -- 找到刚建的菜节点
        FOREACH (step IN dish.steps |                -- 循环做法步骤
            MERGE (s:CookingStep {dish_name: dish.name, order: step.order})  -- 步骤节点：菜名+序号作唯一键
            SET s.order = step.order,
                s.instruction = step.instruction     -- 写入序号和内容
            MERGE (d)-[hs:HAS_STEP]->(s)             -- 关系：菜 -有步骤-> 步骤
            SET hs.order = step.order                -- 关系上也存序号（查询时直接排序）
        )
        """

        # (麻婆豆腐) ──HAS_STEP{order:1}──▶ (CookingStep: 嫩豆腐切块)
        #     │       ──HAS_STEP{order:2}──▶ (CookingStep: 清水加盐煮滚豆腐焯水)
        #     │       ──HAS_STEP{order:3}──▶ (CookingStep: 捞出冲凉沥干水分)
        #     └─────── ──HAS_STEP{order:4}──▶ (CookingStep: 肉末炒干香盛出)
        #             ──HAS_STEP{order:5}──▶ (CookingStep: 豆瓣酱炒出红油) ...

        ingredient_query = """
        UNWIND $batch AS dish
        MATCH (d:Dish {name: dish.name})
        FOREACH (item IN dish.main_ingredients |
            MERGE (i:Ingredient {name: item.name})   -- 食材节点：按名 MERGE，重复菜共用同一食材节点
            MERGE (d)-[rel:HAS_MAIN_INGREDIENT]->(i)
            SET rel.amount_text = item.amount,       -- 关系上存用量："500g"
                rel.role = item.role                 -- 关系上存角色：main
        )
        FOREACH (item IN dish.aux_ingredients |      -- 辅料同样，关系类型换 HAS_AUX_INGREDIENT
            MERGE (i:Ingredient {name: item.name})
            MERGE (d)-[rel:HAS_AUX_INGREDIENT]->(i)
            SET rel.amount_text = item.amount, rel.role = item.role
        )
        """

        # (Dish: 麻婆豆腐)
        #     │──HAS_MAIN_INGREDIENT{amount_text:"2块", role:main}──▶ (Ingredient: 豆腐)
        #     │──HAS_MAIN_INGREDIENT{amount_text:"100g", role:main}──▶ (Ingredient: 猪肉)
        #     │──HAS_AUX_INGREDIENT{amount_text:"1g",   role:aux}──▶ (Ingredient: 花椒)
        #     │──HAS_AUX_INGREDIENT{amount_text:"2个",  role:aux}──▶ (Ingredient: 红辣椒)
        #     │──HAS_AUX_INGREDIENT{amount_text:"1棵",  role:aux}──▶ (Ingredient: 香葱)
        #     │──HAS_AUX_INGREDIENT{amount_text:"1勺",  role:aux}──▶ (Ingredient: 生粉)
        #     └──HAS_AUX_INGREDIENT{amount_text:"5g",   role:aux}──▶ (Ingredient: 植物油) ...

        steps_data = [
            {
                "name": record.name,
                "steps": [
                    {"order": step.order, "instruction": step.instruction}
                    for step in record.steps
                ],
            }
            for record in recipes
        ]
        ingredients_data = [
            {
                "name": record.name,
                "main_ingredients": [
                    {"name": item.name, "amount": item.amount, "role": item.role}
                    for item in record.main_ingredients
                ],
                "aux_ingredients": [
                    {"name": item.name, "amount": item.amount, "role": item.role}
                    for item in record.aux_ingredients
                ],
            }
            for record in recipes
        ]

        for batch in self._chunked(steps_data):
            self._database.execute(step_query, {"batch": batch})

        for batch in self._chunked(ingredients_data):
            self._database.execute(ingredient_query, {"batch": batch})

    # 建 NutritionProfile/HealthBenefit + HAS_NUTRITION_PROFILE/HAS_HEALTH_BENEFIT
    def _attach_ingredient_metadata(self, profiles: List[IngredientProfile]) -> None:
        query = """
        UNWIND $profiles AS profile                -- 扩展每行数据为一行，别名 profile
        MERGE (i:Ingredient {name: profile.name})    -- 找到已有食材节点
        FOREACH (_ IN CASE WHEN profile.nutrition IS NULL THEN [] ELSE [1] END |  -- 有营养才执行
            MERGE (np:NutritionProfile {name: profile.name})
            SET np.description = profile.nutrition
            MERGE (i)-[:HAS_NUTRITION_PROFILE]->(np)
        )
        FOREACH (benefit IN profile.benefits |       -- 每条功效一个节点
            MERGE (hb:HealthBenefit {name: benefit})
            MERGE (i)-[:HAS_HEALTH_BENEFIT]->(hb)
        )
        """

        # (Ingredient: 豆腐)
        #     │──HAS_NUTRITION_PROFILE──▶ (NutritionProfile: 豆腐)
        #     │                              └─ description: "大豆的蛋白质生物学价值可与鱼肉相媲美...
        #     │                                                         大豆蛋白属于完全蛋白质..."
        #     │──HAS_HEALTH_BENEFIT──▶ (HealthBenefit: 抗血栓：大豆中含有的皂苷，清除自由基...)
        #     │──HAS_HEALTH_BENEFIT──▶ (HealthBenefit: 牛奶的替代品：豆腐的营养价值与牛奶接近...)
        #     └──HAS_HEALTH_BENEFIT──▶ (HealthBenefit: ...)

        serialised = [
            {
                "name": profile.name,
                "nutrition": profile.nutrition,
                "benefits": profile.benefits,
            }
            for profile in profiles
        ]

        for batch in self._chunked(serialised):
            self._database.execute(query, {"profiles": batch})

#                     (Flavor: 麻辣) ◀─HAS_FLAVOR─------|
#                    (CookingMethod: 烧) ◀─USES_METHOD─-┤
#                    (DishType: 热菜) ◀─BELONGS_TO_TYPE─┤
#                                                       │
    #    _______________________________________________|
    #   |
# (Dish: 麻婆豆腐) ──HAS_STEP──▶ (CookingStep: 嫩豆腐切块)
#     │──HAS_MAIN_INGREDIENT──▶ (Ingredient: 豆腐) ──HAS_NUTRITION_PROFILE──▶ (NutritionProfile: 豆腐)
#     │                              │──HAS_HEALTH_BENEFIT──▶ (HealthBenefit: 抗血栓...)
#     │                              └──HAS_HEALTH_BENEFIT──▶ (HealthBenefit: 牛奶替代品...)
#     └──HAS_AUX_INGREDIENT──▶ (Ingredient: 花椒) ...
