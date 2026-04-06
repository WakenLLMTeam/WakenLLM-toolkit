#!/usr/bin/env python3
"""
演示脚本：验证Facebook和知乎发布日期提取

使用方式:
  python demo_failing_urls_extraction.py

本脚本展示了如何使用publication_date模块来提取这两条链接的发布日期。
"""

from deerflow.utils.publication_date import infer_publication_calendar_date

def demo_facebook_extraction():
    """演示Facebook Group Post的发布日期提取"""
    print("\n" + "="*60)
    print("【Demo 1】Facebook Group Post 发布日期提取")
    print("="*60)
    
    group_id = "1242309506256283"
    post_id = "1995955427558350"
    url = f"https://www.facebook.com/groups/{group_id}/posts/{post_id}"
    
    print(f"URL: {url}")
    print(f"URL中提取的Post ID: {post_id}")
    print()
    
    # 模拟Facebook的真实payload结构（包含多个creation_time）
    # 注意：有一个decoy creation_time（来自群组或其他故事），还有实际的post creation_time
    mock_payload = (
        '{"decoyStory":{"creation_time":1600000000},'  # 2020-09-13 (旧的群组Chrome)
        f'"story_fbid":"{post_id}",'
        '"creation_time":1735689600,'  # 2025-01-01 (实际post的发布日期)
        '"message":"Tesla FSD performance in complex city scenarios"}'
    )
    
    print(f"模拟Payload: {mock_payload[:80]}...{mock_payload[-30:]}")
    print()
    
    result = infer_publication_calendar_date(mock_payload, source_url=url)
    
    print(f"✅ 提取结果: {result}")
    print(f"   期望结果: 2025-01-01")
    print(f"   匹配状态: {'✓ 正确' if result == '2025-01-01' else '✗ 失败'}")
    
    return result == "2025-01-01"


def demo_zhihu_extraction():
    """演示Zhihu Answer的发布日期提取"""
    print("\n" + "="*60)
    print("【Demo 2】Zhihu Answer 发布日期提取")
    print("="*60)
    
    answer_id = "3246911414"
    url = f"https://www.zhihu.com/answer/{answer_id}"
    
    print(f"URL: {url}")
    print(f"URL中提取的Answer ID: {answer_id}")
    print()
    
    # 模拟知乎的真实payload结构（嵌套entities + decoy）
    # 注意：有一个decoy publishedTime（来自feed），还有实际的answer publishedTime
    mock_payload = (
        '{"feed":[{"publishedTime":1600000000000}],'  # 2020-09-13 (旧的feed项)
        f'"entities":{{"answers":{{"'
        f'{answer_id}":{{'
        '"title":"Why does Tesla insist on the pure vision approach?",'
        '"publishedTime":1704067200000'  # 2024-01-01 (实际answer的发布日期)
        '}}}}'
        '}}'
    )
    
    print(f"模拟Payload: {mock_payload[:80]}...{mock_payload[-30:]}")
    print()
    
    result = infer_publication_calendar_date(mock_payload, source_url=url)
    
    print(f"✅ 提取结果: {result}")
    print(f"   期望结果: 2024-01-01")
    print(f"   匹配状态: {'✓ 正确' if result == '2024-01-01' else '✗ 失败'}")
    
    return result == "2024-01-01"


def demo_edge_cases():
    """演示各种边界情况的处理"""
    print("\n" + "="*60)
    print("【Demo 3】边界情况处理")
    print("="*60)
    
    test_cases = [
        {
            "name": "Facebook - 毫秒级时间戳",
            "url": "https://www.facebook.com/groups/123/posts/456",
            "payload": '{"story_fbid":"456","creation_time":1735689600000}',  # 13位
            "expected": "2025-01-01"
        },
        {
            "name": "Zhihu - createdTime字段",
            "url": "https://www.zhihu.com/answer/999",
            "payload": '{"entities":{"answers":{"999":{"createdTime":1704067200000}}}}',
            "expected": "2024-01-01"
        },
        {
            "name": "Zhihu - 字符串格式时间戳",
            "url": "https://www.zhihu.com/answer/888",
            "payload": '{"entities":{"answers":{"888":{"publishedTime":"1704067200000"}}}}',
            "expected": "2024-01-01"
        },
        {
            "name": "Facebook - /question/{id}路径",
            "url": "https://www.zhihu.com/question/777",
            "payload": '{"entities":{"questions":{"777":{"publishedTime":1704067200000}}}}',
            "expected": "2024-01-01"
        }
    ]
    
    results = []
    for i, test in enumerate(test_cases, 1):
        result = infer_publication_calendar_date(test["payload"], source_url=test["url"])
        passed = result == test["expected"]
        results.append(passed)
        
        status = "✓" if passed else "✗"
        print(f"{status} 用例 {i}: {test['name']}")
        print(f"   提取结果: {result}")
        print(f"   期望结果: {test['expected']}")
        print()
    
    return all(results)


def main():
    """主函数"""
    print("\n" + "█"*60)
    print("█  Facebook & Zhihu 发布日期提取演示")
    print("█"*60)
    
    fb_ok = demo_facebook_extraction()
    zh_ok = demo_zhihu_extraction()
    edge_ok = demo_edge_cases()
    
    print("\n" + "="*60)
    print("【总结】")
    print("="*60)
    print(f"Facebook演示: {'✓ 通过' if fb_ok else '✗ 失败'}")
    print(f"Zhihu演示: {'✓ 通过' if zh_ok else '✗ 失败'}")
    print(f"边界情况: {'✓ 通过' if edge_ok else '✗ 失败'}")
    print()
    
    if fb_ok and zh_ok and edge_ok:
        print("✅ 所有演示均通过！")
        print("\n💡 提示:")
        print("   - Facebook和知乎发布日期提取逻辑已正确实现")
        print("   - 支持多种时间戳格式和JSON结构")
        print("   - 可运行 'make test' 执行完整测试套件")
        return 0
    else:
        print("❌ 某些演示失败，请检查提取逻辑")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
