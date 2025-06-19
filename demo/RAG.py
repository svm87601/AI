
import mysql.connector

# def load_instruction_from_mysql():
#     """Load instruction content from MySQL database"""
#     try:
#         connection = mysql.connector.connect(
#             host="localhost",
#             user="root",
#             password="123456",
#             database="mysql"
#         )
#         cursor = connection.cursor()
#         cursor.execute("SELECT content FROM KnowledgeBase WHERE id = 2")
#         result = cursor.fetchone()

#         if result:
#             return result[0].strip()
#         else:
#             print("Warning: No instruction found for id=2, using default instruction.")
#             return """Answer questions based on the following history and input.
# You are an agricultural expert with advanced knowledge of how crops are grown, and what nutrients are needed.
# Please answer the following agriculture questions"""
        
#     except mysql.connector.Error as err:
#         print(f"Error: {err}")
#         return """Answer questions based on the following history and input.
# You are an agricultural expert with advanced knowledge of how crops are grown, and what nutrients are needed.
# Please answer the following agriculture questions"""
#     finally:
#         if 'cursor' in locals():
#             cursor.close()
#         if 'connection' in locals():
#             connection.close()

def load_instruction_from_mysql():
    sql_path = r"C:\Users\wx\Desktop\All\demo\Mysql\instruction.sql"  # 或者你的实际文件
    try:
        # 最简单：忽略无法解码的字节
        with open(sql_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except FileNotFoundError:
        raise RuntimeError(f"找不到文件：{sql_path}")
    except Exception as e:
        raise RuntimeError(f"读取指令模板失败：{e}")

