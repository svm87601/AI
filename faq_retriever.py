import pymysql
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAQRetriever:
    def __init__(self):
        # 数据库配置
        self.db_config = {
            "host": "localhost",
            "user": "root",
            "password": "123456",
            "database": "faq_database",
            "charset": "utf8mb4",
            "cursorclass": pymysql.cursors.DictCursor
        }
        
        # 加载语义相似度模型
        try:
            self.model = SentenceTransformer(r'C:\Users\cxk\.cache\modelscope\hub\models\sentence-transformers\paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("语义相似度模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
        
        self.similarity_threshold = 0.85  # 相似度阈值
        self.faq_data = self._load_faq_data()
    
    def _load_faq_data(self):
        """从数据库加载FAQ数据"""
        try:
            conn = pymysql.connect(**self.db_config)
            with conn.cursor() as cursor:
                # 获取所有问答对
                cursor.execute("SELECT id, category, question, answer, question_vector FROM faq_data")
                faq_items = cursor.fetchall()
                
                if not faq_items:
                    logger.warning("FAQ数据库中没有数据")
                    return []
                
                # 反序列化向量
                for item in faq_items:
                    if item['question_vector']:
                        try:
                            item['vector'] = np.frombuffer(item['question_vector'], dtype=np.float32)
                        except Exception as e:
                            logger.error(f"向量反序列化失败: {e}")
                            item['vector'] = None
                    else:
                        item['vector'] = None
                
                logger.info(f"成功加载 {len(faq_items)} 条FAQ数据")
                return faq_items
                
        except Exception as e:
            logger.error(f"加载FAQ数据失败: {e}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()
    
    def search_faq(self, query):
        """
        检索FAQ系统
        :param query: 用户问题
        :return: 如果找到匹配项返回答案，否则返回None
        """
        if not self.faq_data:
            logger.warning("FAQ数据为空，无法检索")
            return None
        
        logger.info(f"开始检索FAQ: {query}")
        
        # 1. 先用关键词快速筛选候选集
        keywords = self._extract_keywords(query)
        candidates = self._keyword_search(keywords)
        
        # 如果没有关键词匹配，使用全部数据
        if not candidates:
            candidates = self.faq_data
            logger.info("未找到关键词匹配，将使用全部FAQ数据")
        
        logger.info(f"候选问题数量: {len(candidates)}")
        
        # 2. 计算语义相似度
        try:
            query_vec = self.model.encode([query])[0]
        except Exception as e:
            logger.error(f"问题编码失败: {e}")
            return None
        
        best_match = None
        highest_sim = 0
        
        for item in candidates:
            # 如果发现没有向量的问题，直接生成临时向量使用
            if item.get('vector') is None:
                try:
                    logger.info(f"发现未向量化的问题，ID: {item['id']}，将临时生成向量")
                    item['vector'] = self.model.encode([item['question']])[0]
                except Exception as e:
                    logger.error(f"问题向量生成失败: {e}")
                    continue

            try:
                sim = cosine_similarity([query_vec], [item['vector']])[0][0]
                if sim > highest_sim:
                    highest_sim = sim
                    best_match = item
            except Exception as e:
                logger.error(f"相似度计算失败: {e}")
                continue
        
        logger.info(f"最高相似度: {highest_sim:.4f}")
        
        # 3. 检查是否超过阈值
        if highest_sim >= self.similarity_threshold:
            logger.info(f"找到匹配FAQ: {best_match['question']}")
            return best_match['answer']
        
        logger.info("未找到足够相似的FAQ")
        return None
    
    def _extract_keywords(self, text):
        """关键词提取"""
        if not text:
            return []
        
        # 中文分词简单实现（实际应用中建议使用jieba等分词库）
        stopwords = {"的", "了", "和", "是", "在", "我", "有", "你", "吗", "呢", "？", "怎么", "如何"}
        words = []
        current_word = ""
        
        for char in text:
            if char.strip():
                current_word += char
            else:
                if current_word and current_word not in stopwords:
                    words.append(current_word)
                current_word = ""
        
        if current_word and current_word not in stopwords:
            words.append(current_word)
        
        return words[:5]  # 返回前5个关键词
    
    def _keyword_search(self, keywords):
        """基于关键词的快速检索"""
        if not keywords:
            return []
        
        try:
            conn = pymysql.connect(**self.db_config)
            with conn.cursor() as cursor:
                # 构建关键词搜索条件
                conditions = []
                params = []
                for kw in keywords:
                    conditions.append("(question LIKE %s OR answer LIKE %s)")
                    params.extend([f"%{kw}%", f"%{kw}%"])
                
                sql = f"""
                    SELECT id, category, question, answer, question_vector
                    FROM faq_data
                    WHERE {' OR '.join(conditions)}
                    LIMIT 20
                """
                
                cursor.execute(sql, params)
                results = cursor.fetchall()
                logger.info(f"关键词检索找到 {len(results)} 条结果")
                return results
                
        except Exception as e:
            logger.error(f"关键词检索失败: {e}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()