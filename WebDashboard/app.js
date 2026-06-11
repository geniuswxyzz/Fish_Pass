// 模拟数据生成器，如果未来有后端接口，可替换此部分
function generateMockData(count = 50) {
    const data = [];
    const now = new Date();
    
    for (let i = 0; i < count; i++) {
        const poolId = Math.random() > 0.5 ? 1 : 2;
        const locationType = Math.random() > 0.5 ? 0 : 1; // 0: 起点, 1: 终点
        const flowType = Math.random() > 0.7 ? 1 : 0;     // 0: 上溯, 1: 降河
        
        // 随机生成过去 24 小时内的时间
        const recordTime = new Date(now.getTime() - Math.random() * 24 * 60 * 60 * 1000);
        
        data.push({
            id: count - i, // 倒序 ID
            pool_id: poolId,
            pool_name: `池室 ${poolId}`,
            location_type: locationType,
            flow_type: flowType,
            record_time: recordTime.toLocaleString('zh-CN', { hour12: false }).replace(/\//g, '-')
        });
    }
    
    // 按时间降序排序
    return data.sort((a, b) => new Date(b.record_time) - new Date(a.record_time));
}

let allData = [];
let filteredData = [];
let currentPage = 1;
const itemsPerPage = 10;

// 初始化 ECharts 实例
let poolChart;

document.addEventListener('DOMContentLoaded', () => {
    // 实时时间显示
    const timeDisplay = document.getElementById('current-time');
    setInterval(() => {
        timeDisplay.textContent = new Date().toLocaleString('zh-CN', { hour12: false });
    }, 1000);

    // 初始化图表
    poolChart = echarts.init(document.getElementById('pool-chart'));
    
    // 加载数据并渲染
    // 设置默认日期为今天
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('date-filter').value = today;
    
    loadData();

    // 绑定事件监听
    document.getElementById('pool-filter').addEventListener('change', filterData);
    document.getElementById('date-filter').addEventListener('change', filterData);
    document.getElementById('refresh-btn').addEventListener('click', loadData);
    
    document.getElementById('prev-page').addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            renderTable();
        }
    });
    
    document.getElementById('next-page').addEventListener('click', () => {
        const totalPages = Math.ceil(filteredData.length / itemsPerPage);
        if (currentPage < totalPages) {
            currentPage++;
            renderTable();
        }
    });
});

function loadData() {
    const poolFilter = document.getElementById('pool-filter').value;
    const dateFilter = document.getElementById('date-filter').value;
    
    // 构建请求 URL
    let url = 'http://127.0.0.1:5000/api/logs?';
    if (poolFilter && poolFilter !== 'all') url += `pool_id=${poolFilter}&`;
    if (dateFilter) url += `date=${dateFilter}`;

    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            allData = data;
            filteredData = allData; // 因为后端已经做了筛选，所以直接赋值即可
            currentPage = 1;
            updateDashboard();
            renderTable();
        })
        .catch(error => {
            console.error('Error fetching data:', error);
            // 如果请求失败，可以在页面上给点提示
            const tbody = document.getElementById('table-body');
            tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:red;">获取数据失败，请确保后端服务已启动</td></tr>';
        });
}

function filterData() {
    // 改为重新从后端加载数据
    loadData();
}

function updateDashboard() {
    // 统计逻辑
    let upCount = 0;
    let downCount = 0;
    
    // 简单模拟通过率计算（起点数量 vs 终点数量）
    let startCount1 = 0, endCount1 = 0;
    let startCount2 = 0, endCount2 = 0;

    filteredData.forEach(item => {
        if (item.flow_type === 0) upCount++;
        if (item.flow_type === 1) downCount++;
        
        if (item.pool_id === 1) {
            if (item.location_type === 0) startCount1++;
            if (item.location_type === 1) endCount1++;
        } else {
            if (item.location_type === 0) startCount2++;
            if (item.location_type === 1) endCount2++;
        }
    });

    const total = upCount + downCount;
    
    // 更新卡片数据
    document.getElementById('total-fish').textContent = total;
    document.getElementById('up-count').textContent = upCount;
    document.getElementById('down-count').textContent = downCount;
    
    // 避免除以 0
    const passRate1 = startCount1 === 0 ? 0 : Math.min(100, Math.round((endCount1 / startCount1) * 100));
    const passRate2 = startCount2 === 0 ? 0 : Math.min(100, Math.round((endCount2 / startCount2) * 100));
    
    const avgRate = (passRate1 + passRate2) / (passRate1 > 0 && passRate2 > 0 ? 2 : 1);
    document.getElementById('avg-pass-rate').textContent = `${avgRate.toFixed(1)}%`;

    // 更新图表
    updateChart(passRate1, passRate2);
}

function updateChart(rate1, rate2) {
    const option = {
        tooltip: {
            formatter: '{b}: {c}%'
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            top: '15%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            data: ['池室 1', '池室 2'],
            axisTick: { alignWithLabel: true }
        },
        yAxis: {
            type: 'value',
            max: 100,
            axisLabel: { formatter: '{value} %' }
        },
        series: [
            {
                type: 'bar',
                barWidth: '40%',
                data: [
                    { value: rate1, itemStyle: { color: '#1890ff' } },
                    { value: rate2, itemStyle: { color: '#52c41a' } }
                ],
                label: {
                    show: true,
                    position: 'top',
                    formatter: '{c}%'
                }
            }
        ]
    };
    poolChart.setOption(option);
}

function renderTable() {
    const tbody = document.getElementById('table-body');
    tbody.innerHTML = '';
    
    const totalPages = Math.ceil(filteredData.length / itemsPerPage) || 1;
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = Math.min(startIndex + itemsPerPage, filteredData.length);
    
    const pageData = filteredData.slice(startIndex, endIndex);
    
    if (pageData.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#999;">暂无数据</td></tr>';
    } else {
        pageData.forEach(item => {
            const locBadge = item.location_type === 0 
                ? '<span class="badge start">起点</span>' 
                : '<span class="badge end">终点</span>';
                
            const flowBadge = item.flow_type === 0 
                ? '<span class="badge up">上溯</span>' 
                : '<span class="badge down">降河</span>';

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>#${item.id}</td>
                <td>${item.pool_name}</td>
                <td>${locBadge}</td>
                <td>${flowBadge}</td>
                <td>${item.record_time}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    // 更新分页控件状态
    document.getElementById('page-info').textContent = `第 ${currentPage} 页 / 共 ${totalPages} 页 (共 ${filteredData.length} 条)`;
    document.getElementById('prev-page').disabled = currentPage === 1;
    document.getElementById('next-page').disabled = currentPage === totalPages;
}
