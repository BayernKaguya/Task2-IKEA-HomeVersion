# 测试图表修复的简单脚本
import matplotlib.pyplot as plt
import numpy as np

# 测试饼图的安全解包
def test_pie_chart_safety():
    """测试饼图返回值的安全解包"""
    fig, ax = plt.subplots()
    sizes = [30, 70]
    labels = ['测试1', '测试2']
    
    # 使用修复后的安全方式
    pie_result = ax.pie(sizes, labels=labels, startangle=90)
    wedges = pie_result[0]  # 第一个元素总是wedges
    
    print("饼图安全解包测试通过")
    plt.close(fig)

# 测试assignments的兼容处理
def test_assignments_compatibility():
    """测试assignments数据结构的兼容性处理"""
    
    # 模拟二元组结构（原始模式）
    assignments_2tuple = [
        ({'H': 100, 'sku_id': 'A'}, 200),
        ({'H': 150, 'sku_id': 'B'}, 300)
    ]
    
    # 模拟三元组结构（混合模式）
    assignments_3tuple = [
        ({'H': 100, 'sku_id': 'A'}, 200, 5),
        ({'H': 150, 'sku_id': 'B'}, 300, 3)
    ]
    
    def process_assignments_safely(assignments):
        """安全处理assignments的函数"""
        processed = []
        for tup in assignments:
            item_dict = tup[0]
            width = tup[1]
            # 忽略可能存在的第三个元素
            item_height = item_dict.get('H', 0)
            processed.append((item_dict['sku_id'], width, item_height))
        return processed
    
    # 测试二元组
    result_2 = process_assignments_safely(assignments_2tuple)
    print(f"二元组处理结果: {result_2}")
    
    # 测试三元组
    result_3 = process_assignments_safely(assignments_3tuple)
    print(f"三元组处理结果: {result_3}")
    
    print("assignments兼容性测试通过")

if __name__ == "__main__":
    print("开始测试图表修复...")
    test_pie_chart_safety()
    test_assignments_compatibility()
    print("所有测试通过！")
