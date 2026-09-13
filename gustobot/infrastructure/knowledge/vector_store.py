"""
Milvus向量数据库封装
使用Milvus作为向量存储引擎
"""
from typing import List, Dict, Any, Optional
from loguru import logger
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)


class VectorStore:
    """Milvus向量数据库管理类"""

    def __init__(
        self,
        collection_name: str = "recipes",
        host: str = "localhost",
        port: int = 19530,
        dimension: int = 1536,
        index_type: str = "IVF_FLAT",
        metric_type: str = "IP"  # Inner Product (cosine similarity)
    ):
        """
        初始化Milvus向量数据库

        Args:
            collection_name: 集合名称
            host: Milvus服务器地址
            port: Milvus服务器端口
            dimension: 向量维度
            index_type: 索引类型 (IVF_FLAT, IVF_SQ8, HNSW等)
            metric_type: 距离度量类型 (IP, L2, COSINE)
        """
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.dimension = dimension
        self.index_type = index_type
        self.metric_type = metric_type

        self.collection = None

        self._initialize()

    def _initialize(self):
        """初始化Milvus连接和集合"""
        try:
            # 连接到Milvus
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")

            # 检查集合是否存在，若不存在创造新集合
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            else:
                # 创建新集合
                self._create_collection()
                logger.info(f"Created new collection: {self.collection_name}")

            # 加载集合到内存
            self.collection.load()

        except Exception as e:
            logger.error(f"Failed to initialize Milvus: {e}")
            raise

    def _create_collection(self):
        """创建Milvus集合"""
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),  # 原文
            FieldSchema(name="recipe_id", dtype=DataType.VARCHAR, max_length=256),  # 来源文档
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=512),  # 标题
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),  # 分类
            FieldSchema(name="difficulty", dtype=DataType.VARCHAR, max_length=128),
        ]

        # 创建集合schema
        schema = CollectionSchema(
            fields=fields,
            description="Recipe knowledge base collection"
        )

        # 创建集合
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )

        # 创建索引
        index_params = {
            "index_type": self.index_type,  # IVF_FLAT
            "metric_type": self.metric_type,  # IP, embedding 做了 L2 归一化，归一化后 IP ≡ COSINE，但 IP 计算更快
            "params": {"nlist": 128} if self.index_type == "IVF_FLAT" else {}  # IVF 聚类数
        }

        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        logger.info(f"Created index with type: {self.index_type}")

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        添加文档到向量数据库

        Args:
            ids: 文档ID列表
            embeddings: 嵌入向量列表
            documents: 文档文本列表
            metadatas: 文档元数据列表

        Returns:
            是否成功
        """
        try:
            if not metadatas:
                metadatas = [{}] * len(ids)

            def _as_varchar(value: Any) -> str:
                if value is None:
                    return ""
                return str(value)

            # 准备插入数据
            entities = [
                ids,
                embeddings,
                documents,
                [_as_varchar(meta.get("recipe_id")) for meta in metadatas],
                [_as_varchar(meta.get("name")) for meta in metadatas],
                [_as_varchar(meta.get("category")) for meta in metadatas],
                [_as_varchar(meta.get("difficulty")) for meta in metadatas],
            ]

            # [['history_001_0'], [[0.013065679930150509,..., -0.03169622644782066]], ['在探讨古代饮食文化时...就得先知 道古代有什么吃的。'], ['history_001'], ['中国饮食文化历史资料 - 第1段'], ['历史文化'], ['']]
            # print("-------------------------------")
            # print(entities)

            # 插入数据
            self.collection.insert(entities)
            self.collection.flush()  # 强制写入磁盘

            logger.info(f"Added {len(ids)} documents to Milvus")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似文档

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            filter_expr: 过滤表达式 (例如: "category == '家常菜'")

        Returns:
            搜索结果列表
        """
        try:
            # 搜索参数
            search_params = {
                "metric_type": self.metric_type,  # 相似度度量方式
                "params": {"nprobe": 10}  # 搜索精度参数，nprobe：搜索时检查多少个聚类，值越大，搜索越精确，但速度越慢
            }

            # 执行搜索
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["id", "content", "recipe_id", "name", "category", "difficulty"]  # 返回哪些字段
            )

            # 格式化结果
            formatted_results = []
            print("*******************************************************")
            print(results)
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "id": hit.entity.get("id"),
                        "content": hit.entity.get("content"),
                        "score": float(hit.score),
                        "metadata": {
                            "recipe_id": hit.entity.get("recipe_id"),
                            "name": hit.entity.get("name"),
                            "category": hit.entity.get("category"),
                            "difficulty": hit.entity.get("difficulty"),
                        }
                    })

            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def delete_documents(self, ids: List[str]) -> bool:
        """
        删除文档

        Args:
            ids: 文档ID列表

        Returns:
            是否成功
        """
        try:
            expr = f"id in {ids}"
            self.collection.delete(expr)
            self.collection.flush()

            logger.info(f"Deleted {len(ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            stats = self.collection.num_entities
            return {
                "name": self.collection_name,
                "document_count": stats,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric_type": self.metric_type
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def clear_collection(self) -> bool:
        """清空集合"""
        try:
            self.collection.drop()
            self._create_collection()
            self.collection.load()

            logger.info(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def close(self):
        """关闭连接"""
        try:
            connections.disconnect("default")
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Failed to disconnect: {e}")
