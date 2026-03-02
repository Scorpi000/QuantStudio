import json
import time
import logging
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import Field, BaseModel

from QuantStudio.Core import __QS_Object__


logger = logging.getLogger()

class Message(BaseModel):
    """消息类，用于顶点间通信"""
    source_id: int = Field(title="源顶点ID")
    target_id: int = Field(title="目标顶点ID")
    value: Any = Field(title="消息值")
    superstep: int = 0

    def __repr__(self):
        return f"Message({self.source_id}->{self.target_id}: {self.value})"

class MessageBuffer:
    """消息缓冲区，用于管理和路由消息"""

    def __init__(self):
        self._messages: Dict[Any, List[Message]] = defaultdict(list)
        self._lock = None

    def init_lock(self, lock):
        """初始化锁"""
        self._lock = lock

    def add_message(self, message: Message):
        """添加一条消息到缓冲区"""
        with self._lock:
            self._messages[message.target_id].append(message)

    def add_messages(self, messages: List[Message]):
        """批量添加消息"""
        with self._lock:
            for msg in messages:
                self._messages[msg.target_id].append(msg)

    def get_messages_for_vertex(self, vertex_id: Any) -> List[Message]:
        """获取发送给指定顶点的所有消息"""
        with self._lock:
            messages = self._messages.get(vertex_id, [])
            self._messages[vertex_id] = []
            return messages

    def has_messages_for_vertex(self, vertex_id: Any) -> bool:
        """检查是否有发送给指定顶点的消息"""
        with self._lock:
            return len(self._messages.get(vertex_id, [])) > 0

    def clear(self):
        """清空缓冲区"""
        with self._lock:
            self._messages.clear()

    def __len__(self):
        with self._lock:
            return sum(len(msgs) for msgs in self._messages.values())

class Aggregator(ABC):
    """聚合器基类，用于全局统计和信息收集"""

    def __init__(self, name: str):
        self.name = name
        self._value = None
        self._lock = None  # 将在init_lock中初始化

    def init_lock(self, lock):
        """初始化锁"""
        self._lock = lock

    @abstractmethod
    def aggregate(self, value: Any):
        pass

    @abstractmethod
    def finalize(self) -> Any:
        pass

    @abstractmethod
    def reset(self):
        pass

    def get_value(self) -> Any:
        with self._lock:
            return self._value

class Vertex(ABC):
    """顶点抽象基类"""

    def __init__(self, vertex_id: Any):
        self.id = vertex_id
        self.value = None
        self.edges: List[tuple] = []
        self.active = True
        self.superstep = 0
        self._worker = None
        self._message_buffer = []

    def set_worker(self, worker):
        self._worker = worker

    def add_edge(self, neighbor_id: Any, weight: Any = None):
        if weight is None:
            self.edges.append((neighbor_id, 1))
        else:
            self.edges.append((neighbor_id, weight))

    def get_neighbors(self) -> List[Any]:
        return [edge[0] for edge in self.edges]

    def send_message(self, target_id: Any, value: Any):
        """发送消息给目标顶点"""
        if self._worker:
            message = Message(
                target_id=target_id,
                value=value,
                source_id=self.id,
                superstep=self.superstep
            )
            self._worker.send_message(message)
        else:
            self._message_buffer.append(Message(
                target_id=target_id,
                value=value,
                source_id=self.id,
                superstep=self.superstep
            ))

    def send_message_to_all(self, value: Any):
        """向所有邻居顶点发送相同的消息"""
        for neighbor_id, _ in self.edges:
            self.send_message(neighbor_id, value)

    def vote_to_halt(self):
        """投票停止当前顶点的计算"""
        self.active = False

    def activate(self):
        """激活顶点"""
        self.active = True

    def get_incoming_messages(self) -> List[Message]:
        """获取当前超步收到的所有消息"""
        if self._worker:
            return self._worker.get_messages(self.id)
        messages = self._message_buffer.copy()
        self._message_buffer.clear()
        return messages

    @abstractmethod
    def compute(self, messages: List[Message]):
        """顶点计算逻辑（必须由子类实现）"""
        raise NotImplementedError

    def __repr__(self):
        return f"Vertex(id={self.id}, active={self.active}, value={self.value})"

class Graph:
    """图数据结构，管理顶点和边"""

    def __init__(self):
        self.vertices: Dict[Any, Vertex] = {}
        self.edges: List[tuple] = []

    def add_vertex(self, vertex: Vertex):
        self.vertices[vertex.id] = vertex

    def add_edge(self, source_id: Any, target_id: Any, weight: Any = 1):
        self.vertices[source_id].add_edge(target_id, weight)
        self.edges.append((source_id, target_id, weight))

    def from_edge_list(self, edges: List[tuple]):
        for edge in edges:
            if len(edge) == 2:
                source, target = edge
                weight = 1
            else:
                source, target, weight = edge
            self.add_edge(source, target, weight)

    def from_adjacency_list(self, adjacency_list: Dict[Any, List[tuple]]):
        for vertex_id, neighbors in adjacency_list.items():
            for neighbor_info in neighbors:
                if isinstance(neighbor_info, tuple):
                    neighbor_id, weight = neighbor_info
                else:
                    neighbor_id = neighbor_info
                    weight = 1
                self.add_edge(vertex_id, neighbor_id, weight)

    def get_vertex_count(self) -> int:
        return len(self.vertices)

    def get_edge_count(self) -> int:
        return len(self.edges)

    def get_vertex(self, vertex_id: Any) -> Optional[Vertex]:
        return self.vertices.get(vertex_id)

    def get_all_vertices(self) -> List[Vertex]:
        return list(self.vertices.values())

class Worker:
    """Worker工作节点（单线程版本）"""

    def __init__(self, worker_id: int, vertices: List[Vertex], num_workers: int = 1):
        self.worker_id = worker_id
        self.vertices = {v.id: v for v in vertices}
        self.num_workers = num_workers
        self.message_buffer = MessageBuffer()
        self.message_buffer.init_lock(lock=__import__('threading').Lock())
        self.outgoing_messages: List[Message] = []
        self.current_superstep = 0
        self.aggregators: Dict[str, Aggregator] = {}

        # 设置顶点关联
        for vertex in self.vertices.values():
            vertex.set_worker(self)

        # 初始化聚合器锁
        for agg in self.aggregators.values():
            agg.init_lock(__import__('threading').Lock())

    def register_aggregator(self, aggregator: Aggregator):
        self.aggregators[aggregator.name] = aggregator
        aggregator.init_lock(__import__('threading').Lock())

    def send_message(self, message: Message):
        """发送消息（内部方法）"""
        self.outgoing_messages.append(message)

    def get_messages(self, vertex_id: Any) -> List[Message]:
        """获取顶点的消息"""
        return self.message_buffer.get_messages_for_vertex(vertex_id)

    def route_messages(self):
        """路由消息到目标顶点"""
        for message in self.outgoing_messages:
            self.message_buffer.add_message(message)
        self.outgoing_messages.clear()

    def reset_aggregators(self):
        for aggregator in self.aggregators.values():
            aggregator.reset()

    def run_compute(self) -> tuple:
        """执行一个超步的计算"""
        active_count = 0

        for vertex_id, vertex in self.vertices.items():
            vertex.superstep = self.current_superstep
            messages = self.message_buffer.get_messages_for_vertex(vertex_id)

            if not vertex.active and messages:
                vertex.activate()

            if vertex.active or messages:
                try:
                    vertex.compute(messages)
                    if vertex.active:
                        active_count += 1
                except Exception as e:
                    logger.error(f"Error in vertex {vertex_id} compute: {e}")
                    raise

        self.route_messages()
        return active_count, len(self.outgoing_messages) + len(self.message_buffer)

    def save_checkpoint(self) -> dict:
        """保存检查点"""
        checkpoint_data = {
            'superstep': self.current_superstep,
            'vertices': {}
        }
        for vertex_id, vertex in self.vertices.items():
            checkpoint_data['vertices'][vertex_id] = {
                'value': vertex.value,
                'active': vertex.active,
                'edges': vertex.edges
            }
        return checkpoint_data

    def load_checkpoint(self, checkpoint_data: dict):
        """加载检查点"""
        self.current_superstep = checkpoint_data['superstep']
        for vertex_id, data in checkpoint_data['vertices'].items():
            if vertex_id in self.vertices:
                self.vertices[vertex_id].value = data['value']
                self.vertices[vertex_id].active = data['active']
                self.vertices[vertex_id].edges = data['edges']

class Master:
    """Master主节点（简化版）"""

    def __init__(self, graph: Graph, num_workers: int = 1, checkpoint_interval: int = 10):
        self.graph = graph
        self.num_workers = num_workers if num_workers > 0 else 1
        self.checkpoint_interval = checkpoint_interval

        # 创建Worker并分区图
        self.workers: List[Worker] = []
        self._create_workers()

        # 消息中转
        self.global_aggregators: Dict[str, Any] = {}
        self.current_superstep = 0
        self.halt_sent = False

    def _create_workers(self):
        """创建Worker并分区图"""
        for worker_id in range(self.num_workers):
            partition = self._get_partition(worker_id)
            worker = Worker(worker_id, partition, self.num_workers)
            self.workers.append(worker)
            logger.info(f"Created Worker {worker_id} with {len(partition)} vertices")

    def _get_partition(self, worker_id: int) -> List[Vertex]:
        """获取分配给指定worker的顶点分区"""
        partition = []
        for vertex_id, vertex in self.graph.vertices.items():
            if hash(str(vertex_id)) % self.num_workers == worker_id:
                partition.append(vertex)
        return partition

    def run_superstep(self) -> bool:
        """执行一个超步"""
        logger.info(f"Starting Superstep {self.current_superstep}")

        # 重置聚合器
        for worker in self.workers:
            worker.reset_aggregators()

        # 执行计算
        total_active = 0
        for worker in self.workers:
            active_count, _ = worker.run_compute()
            total_active += active_count

        self.current_superstep += 1

        logger.info(f"Superstep {self.current_superstep - 1}: active={total_active}")

        return total_active > 0

    def run(self, max_supersteps: int = 100) -> dict:
        """运行完整的Pregel计算"""
        start_time = time.time()
        stats = {
            'total_supersteps': 0,
            'final_active_vertices': 0,
            'execution_time': 0
        }

        try:
            for step in range(max_supersteps):
                if not self.run_superstep():
                    logger.info("All vertices halted. Terminating.")
                    break

            stats['total_supersteps'] = self.current_superstep

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        execution_time = time.time() - start_time
        stats['execution_time'] = execution_time

        logger.info(f"Execution completed in {execution_time:.2f} seconds")
        logger.info(f"Total supersteps: {stats['total_supersteps']}")

        return stats

    def get_vertex_values(self) -> Dict[Any, Any]:
        """获取所有顶点的最终值"""
        return {v.id: v.value for v in self.graph.get_all_vertices()}

    def save_checkpoint(self) -> dict:
        """保存全局检查点"""
        checkpoint = {}
        for worker in self.workers:
            checkpoint[worker.worker_id] = worker.save_checkpoint()
        return checkpoint

    def load_checkpoint(self, checkpoint: dict):
        """从检查点恢复"""
        for worker in self.workers:
            if worker.worker_id in checkpoint:
                worker.load_checkpoint(checkpoint[worker.worker_id])

class PregelEngine:
    """Pregel计算引擎"""

    def __init__(self, graph: Graph, num_workers: int = 1, checkpoint_interval: int = 10):
        self.graph = graph
        self.num_workers = num_workers
        self.checkpoint_interval = checkpoint_interval
        self.master = None

    def run(self, max_supersteps: int = 100) -> dict:
        """运行Pregel计算"""
        logger.info("Starting Pregel computation...")

        self.master = Master(
            graph=self.graph,
            num_workers=self.num_workers,
            checkpoint_interval=self.checkpoint_interval
        )

        stats = self.master.run(max_supersteps=max_supersteps)
        return stats

    def get_vertex_values(self) -> Dict[Any, Any]:
        """获取所有顶点的最终值"""
        if self.master:
            return self.master.get_vertex_values()
        return {}

    def save_checkpoint(self) -> dict:
        """保存检查点"""
        if self.master:
            return self.master.save_checkpoint()
        return {}

    def load_checkpoint(self, checkpoint: dict):
        """从检查点恢复"""
        if self.master:
            self.master.load_checkpoint(checkpoint)