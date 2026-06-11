#pragma once
#include <string>
#include <memory>
#include <mutex>

// 引入 MySQL 驱动头文件 (请确保已配置环境)
#include <mysql_driver.h>
#include <mysql_connection.h>
#include <cppconn/prepared_statement.h>

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
            std::cerr << "数据库连接失败: " << e.what() << std::endl;
            return false;
        }
    }

    // 核心：插入流水记录
    void InsertPassageLog(int pool_id, int location_type, int flow_type) {
        if (!con) return;

        try {
            std::unique_lock<std::mutex> lock(stmtMutex); // 线程安全锁

            // 预编译 SQL (性能高且防注入)
            // pass_time 直接使用数据库函数 NOW()
            std::unique_ptr<sql::PreparedStatement> pstmt(con->prepareStatement(
                "INSERT INTO fish_passage_log (pool_id, location_type, flow_type, pass_time) VALUES (?, ?, ?, NOW())"
            ));

            pstmt->setInt(1, pool_id);
            pstmt->setInt(2, location_type);
            pstmt->setInt(3, flow_type);

            pstmt->executeUpdate();

            // 可选：打印日志
            // std::cout << "[DB Success] Inserted log for Pool " << pool_id << std::endl;
        }
        catch (sql::SQLException& e) {
            std::cerr << "SQL 插入错误: " << e.what() << std::endl;
            // 实际工程中这里可能需要重连逻辑
        }
    }

private:
    DBHelper() {} // 私有构造
    ~DBHelper() {
        if (con) con->close();
    }

    // 禁止拷贝
    DBHelper(const DBHelper&) = delete;
    DBHelper& operator=(const DBHelper&) = delete;

    sql::mysql::MySQL_Driver* driver = nullptr;
    std::unique_ptr<sql::Connection> con;
    std::mutex stmtMutex; // 保护数据库并发写入
};