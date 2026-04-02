"""
Pregel算法实现示例

包含以下经典图计算算法：
1. MaxValue - 传播并计算图中的最大值
2. PageRank - 网页排名算法
3. SingleSourceShortestPath (SSSP) - 单源最短路径
4. ConnectedComponents - 连通分量
5. BFS - 广度优先搜索
"""

import sys
import os
from typing import Any

from QuantStudio.Core.PregelEngine import Vertex, Message, Aggregator


class SumAggregator(Aggregator):
    """求和聚合器"""

    def __init__(self, name: str):
        super().__init__(name)

    def aggregate(self, value: Any):
        if not hasattr(self, '_initialized'):
            self.init_lock(__import__('threading').Lock())
            self._initialized = True
        with self._lock:
            if isinstance(value, (int, float)):
                self._value = (self._value or 0) + value

    def finalize(self) -> Any:
        with self._lock:
            return self._value

    def reset(self):
        with self._lock:
            self._value = 0


class MaxAggregator(Aggregator):
    """最大值聚合器"""

    def __init__(self, name: str):
        super().__init__(name)

    def aggregate(self, value: Any):
        if not hasattr(self, '_initialized'):
            self.init_lock(__import__('threading').Lock())
            self._initialized = True
        with self._lock:
            if self._value is None or value > self._value:
                self._value = value

    def finalize(self) -> Any:
        with self._lock:
            return self._value

    def reset(self):
        with self._lock:
            self._value = None


class CountAggregator(Aggregator):
    """计数聚合器"""

    def __init__(self, name: str):
        super().__init__(name)
        self.reset()

    def aggregate(self, value: Any = 1):
        with self._lock:
            self._value = (self._value or 0) + value

    def finalize(self) -> Any:
        with self._lock:
            return self._value

    def reset(self):
        with self._lock:
            self._value = 0


# ============================================================================
# 1. MaxValue算法
# ============================================================================

class MaxValueVertex(Vertex):
    """
    MaxValue算法实现

    功能：找到图中所有可达顶点中的最大值

    算法原理：
    1. 初始化时设置顶点值为自身ID（或指定初始值）
    2. 每个超步中，顶点比较自身值和收到的消息值
    3. 如果收到更大的值，更新自身并广播给所有邻居
    4. 当一个超步中没有更新发生时，所有顶点投票停止

    使用场景：
    - 查找图中的最大值
    - 传播广播消息
    - 测试Pregel框架的基本功能
    """

    def __init__(self, vertex_id: Any, initial_value: Any = None):
        super().__init__(vertex_id)
        self.first_run = True
        # 如果没有指定初始值，使用顶点ID作为初始值
        self.value = initial_value if initial_value is not None else vertex_id

    def compute(self, messages: list):
        """
        计算逻辑：
        1. 比较当前值和消息中的最大值
        2. 如果有更新，发送新值给所有邻居
        3. 如果没有更新，投票停止
        """
        # 获取当前最大值
        current_max = self.value

        # 处理收到的消息，找出最大值
        for msg in messages:
            if isinstance(msg.value, (int, float)):
                if msg.value > current_max:
                    current_max = msg.value

        # 如果有更新
        if self.first_run or (current_max != self.value):
            self.value = current_max
            # 向所有邻居发送新值
            self.send_message_to_all(current_max)
            self.first_run = False
        else:
            # 没有更新，投票停止
            self.vote_to_halt()


if __name__=="__main__":
    # 创建一个图，其中一些顶点有较大的值，
    # 验证最大值是否能够正确传播到所有可达顶点

    from QuantStudio.Core.PregelEngine import Graph, PregelEngine

    # 创建测试图
    graph = Graph()

    # 添加顶点，设定初始值
    vertices_data = {
        1: 10,   # 最大值
        2: 5,
        3: 3,
        4: 8,
        5: 2,
        6: 7,
        7: 1,
        8: 6,
        9: 4,
        10: 9
    }

    for vid, value in vertices_data.items():
        graph.add_vertex(MaxValueVertex(vid, initial_value=value))

    # 添加边
    edges = [
        (1, 2), (1, 3),
        (2, 4), (2, 5),
        (3, 6), (3, 7),
        (4, 8), (4, 9),
        (5, 9), (5, 10),
        (6, 8), (6, 10),
        (7, 9),
        (8, 10)
    ]

    for src, tgt in edges:
        graph.add_edge(src, tgt)

    print("=" * 60)
    print("MaxValue算法示例")
    print("=" * 60)
    print(f"图信息: {graph}")
    print(f"初始值: {vertices_data}")
    print("预期结果: 所有顶点值应为10（最大值）")
    print("-" * 60)

    # 运行计算
    engine = PregelEngine(graph, num_workers=1)
    stats = engine.run(max_supersteps=20)

    # 获取结果
    results = engine.get_vertex_values()
    print(f"最终结果: {results}")

    # 验证结果
    expected_value = max(vertices_data.values())
    all_correct = all(v == expected_value for v in results.values())
    print(f"验证通过: {all_correct}")
    print("-" * 60)