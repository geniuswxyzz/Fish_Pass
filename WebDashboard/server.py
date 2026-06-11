from flask import Flask, jsonify, request
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)
CORS(app)

# 数据库配置（与 C++ 项目中的 DBHelper 保持一致）
DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '123',
    'database': 'fish_db'
}

def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None

@app.route('/api/logs', methods=['GET'])
def get_logs():
    pool_id = request.args.get('pool_id')
    date = request.args.get('date')
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Failed to connect to database"}), 500
        
    try:
        cursor = conn.cursor(dictionary=True)
        
        query = "SELECT id, pool_id, location_type, flow_type, pass_time as record_time FROM fish_passage_log WHERE 1=1"
        params = []
        
        if pool_id and pool_id != 'all':
            query += " AND pool_id = %s"
            params.append(int(pool_id))
            
        if date:
            query += " AND DATE(pass_time) = %s"
            params.append(date)
            
        query += " ORDER BY pass_time DESC, id DESC"
        
        cursor.execute(query, params)
        records = cursor.fetchall()
        
        # 格式化时间，并将名称等附加信息加进去
        for r in records:
            if r['record_time']:
                r['record_time'] = r['record_time'].strftime('%Y-%m-%d %H:%M:%S')
            r['pool_name'] = f"池室 {r['pool_id']}"
            
        return jsonify(records)
        
    except Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == '__main__':
    print("Starting Flask server for FishPass Dashboard on port 5000...")
    app.run(debug=True, port=5000)
