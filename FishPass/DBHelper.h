#pragma once
#include <string>
#include <memory>
#include <mutex>
#include <iostream>

// 引入 MySQL 驱动头文件
#include <mysql_driver.h>
#include <mysql_connection.h>
#include <cppconn/prepared_statement.h>
#include <cppconn/statement.h>
#include <cppconn/resultset.h>

// [Code God] 核心修复：定义统计结果结构体
// 必须定义在 DBHelper 类外面或者 public 区域，否则 main.cpp 无法使用
struct PoolStats {
    int start_net_flow; // 起点净流量 (上溯 - 降河)
    int end_net_flow;   // 终点净流量
    double pass_rate;   // 通过率 (0.0 - 1.0)
    bool is_valid;      // 数据是否有效
};

class DBHelper {
public:
    // 获取单例实例
    static DBHelper& GetInstance() {
        static DBHelper instance;
        return instance;
    }

    // 初始化连接
    bool Connect(const std::string& host, const std::string& user, const std::string& pass, const std::string& db) {
        try {
            driver = sql::mysql::get_mysql_driver_instance();
            con.reset(driver->connect(host, user, pass));
            con->setSchema(db);
            return true;
        }
        catch (sql::SQLException& e) {
            std::cerr << "[DB Error] 连接失败: " << e.what() << std::endl;
            return false;
        }
    }

    // [Code God] 插入数据，并强制保持只存最新的 100 条
    void InsertPassageLog(int pool_id, int location_type, int flow_type) {
        if (!con) return;
        try {
            // 1. 正常的插入逻辑 (保持不变)
            std::unique_lock<std::mutex> lock(stmtMutex);
            std::unique_ptr<sql::PreparedStatement> pstmt(con->prepareStatement(
                "INSERT INTO fish_passage_log (pool_id, location_type, flow_type, pass_time) VALUES (?, ?, ?, NOW())"
            ));
            pstmt->setInt(1, pool_id);
            pstmt->setInt(2, location_type);
            pstmt->setInt(3, flow_type);
            pstmt->executeUpdate();

            // =========================================================
            // [Code God] 新增：自动清理逻辑 (Rolling Buffer)
            // 逻辑：如果数据量超过 100，就删除最旧的数据
            // =========================================================

            // 设定最大保留条数
            const int MAX_LOG_SIZE = 100;

            // 创建一个 Statement 执行清理 SQL
            std::unique_ptr<sql::Statement> stmt(con->createStatement());

            // 这句 SQL 的含义是：
            // 1. 找出倒数第 100 条数据的 ID (Subquery)
            // 2. 删除所有 ID 小于该 ID 的记录 (即更早的记录)
            // 注意：MySQL 不允许在 Delete 的 Where 子句直接查同一张表，所以要嵌套一层 (SELECT ... AS t)
            std::string cleanup_sql =
                "DELETE FROM fish_passage_log WHERE id < ("
                "   SELECT min_id FROM ("
                "       SELECT id AS min_id FROM fish_passage_log "
                "       ORDER BY id DESC LIMIT 1 OFFSET " + std::to_string(MAX_LOG_SIZE - 1) +
                "   ) AS t"
                ")";

            // 执行清理
            stmt->executeUpdate(cleanup_sql);
        }
        catch (sql::SQLException& e) {
            std::cerr << "[DB Insert/Cleanup Error] " << e.what() << std::endl;
        }
    }

    // [Code God] 新增：按时间段查询通过率
    PoolStats QueryPassRateByTime(int pool_id, std::string start_time, std::string end_time) {
        PoolStats stats = { 0, 0, 0.0, false };
        if (!con) return stats;
        // 【重点】这里必须加锁！
        // 因为主线程正在 Insert，如果你这里不加锁直接用 con 创建 Statement，会发生竞争冲突
        std::unique_lock<std::mutex> lock(stmtMutex);
        try {
            std::unique_ptr<sql::Statement> stmt(con->createStatement());

            // 这里的 SQL 逻辑：分别计算起点净流量和终点净流量
            std::string sql =
                "SELECT "
                "  (SUM(CASE WHEN location_type=0 AND flow_type=0 THEN 1 ELSE 0 END) - "
                "   SUM(CASE WHEN location_type=0 AND flow_type=1 THEN 1 ELSE 0 END)) as start_net, "
                "  (SUM(CASE WHEN location_type=1 AND flow_type=0 THEN 1 ELSE 0 END) - "
                "   SUM(CASE WHEN location_type=1 AND flow_type=1 THEN 1 ELSE 0 END)) as end_net "
                "FROM fish_passage_log "
                "WHERE pool_id = " + std::to_string(pool_id) +
                " AND pass_time BETWEEN '" + start_time + "' AND '" + end_time + "'";

            std::unique_ptr<sql::ResultSet> res(stmt->executeQuery(sql));

            if (res->next()) {
                stats.start_net_flow = res->getInt("start_net");
                stats.end_net_flow = res->getInt("end_net");

                // 分母不为0才能计算通过率
                if (stats.start_net_flow != 0) {
                    // 1. 先算出原始倍数 (例如 1.25)
                    double raw_rate = (double)stats.end_net_flow / (double)stats.start_net_flow;

                    // 2. [Code God 补丁] 加上你“消失”的修正逻辑
                    if (raw_rate > 1.0) {
                        stats.pass_rate = 1.0;  // 强制设为 100%
                    }
                    else if (raw_rate < 0) {
                        stats.pass_rate = 0.0;  // 防止出现负数
                    }
                    else {
                        stats.pass_rate = raw_rate; // 正常情况
                    }

                    stats.is_valid = true;
                }
                else {
                    stats.pass_rate = 0.0;
                    stats.is_valid = false;
                }
            }
        }
        catch (sql::SQLException& e) {
            std::cerr << "[DB Query Error] " << e.what() << std::endl;
        }
        return stats;
    }

    // [Code God] 新增：查询指定时间段内的详细流水记录
    void QueryLogsByTime(int pool_id, std::string start_time, std::string end_time) {
        if (!con) return;

        // 【必须加锁】防止查询时主线程插入数据导致崩溃
        std::unique_lock<std::mutex> lock(stmtMutex);

        try {
            std::unique_ptr<sql::Statement> stmt(con->createStatement());

            std::string sql =
                "SELECT * FROM fish_passage_log "
                "WHERE pool_id = " + std::to_string(pool_id) +
                " AND pass_time BETWEEN '" + start_time + "' AND '" + end_time + "'"
                " ORDER BY pass_time DESC"; // 按时间倒序，最新的在前面

            std::unique_ptr<sql::ResultSet> res(stmt->executeQuery(sql));

            std::cout << "  >>> 详细记录 (Pool ID: " << pool_id << ") <<<" << std::endl;
            std::cout << "  --------------------------------------------------" << std::endl;

            bool has_data = false;
            while (res->next()) {
                has_data = true;
                int loc = res->getInt("location_type");
                int flow = res->getInt("flow_type");

                std::string loc_str = (loc == 0 ? "起点" : "终点");
                // 结合之前的逻辑：flow=0是上溯，flow=1是降河
                std::string flow_str = (flow == 0 ? "上溯" : "降河");

                std::cout << "  [ID:" << res->getInt("id") << "] "
                    << "时间: " << res->getString("pass_time") << " | "
                    << loc_str << " | " << flow_str << std::endl;
            }

            if (!has_data) {
                std::cout << "  (该时间段内无记录)" << std::endl;
            }
            std::cout << "  --------------------------------------------------\n" << std::endl;
        }
        catch (sql::SQLException& e) {
            std::cerr << "[DB Query Log Error] " << e.what() << std::endl;
        }
    }
    // 调试用：打印最近记录
    void DebugQuery() {
        if (!con) return;
        try {
            std::unique_ptr<sql::Statement> stmt(con->createStatement());
            std::unique_ptr<sql::ResultSet> res(stmt->executeQuery("SELECT * FROM fish_passage_log ORDER BY id DESC LIMIT 5"));

            std::cout << "\n=== [DB Debug] 最近流水记录 ===" << std::endl;
            while (res->next()) {
                std::cout << "ID: " << res->getInt("id")
                    << " | Pool: " << res->getInt("pool_id")
                    << " | Loc: " << (res->getInt("location_type") == 0 ? "起点" : "终点")
                    << " | Flow: " << (res->getInt("flow_type") == 0 ? "上溯" : "降河")
                    << " | Time: " << res->getString("pass_time") << std::endl;
            }
            std::cout << "===============================\n" << std::endl;
        }
        catch (sql::SQLException& e) {
            std::cerr << "[DB Debug Error] " << e.what() << std::endl;
        }
    }

private:
    DBHelper() {}
    ~DBHelper() { if (con) con->close(); }
    DBHelper(const DBHelper&) = delete;
    DBHelper& operator=(const DBHelper&) = delete;

    sql::mysql::MySQL_Driver* driver = nullptr;
    std::unique_ptr<sql::Connection> con;
    std::mutex stmtMutex;
};