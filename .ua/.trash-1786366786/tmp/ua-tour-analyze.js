/**
 * QuantStudio 项目拓扑分析脚本
 *
 * 读取项目图数据（节点和边），执行以下分析：
 * A. Fan-In 排名（重要性）
 * B. Fan-Out 排名（作用范围）
 * C. 入口点候选评分
 * D. BFS 依赖链遍历
 * E. 非代码文件清单
 * F. 紧密耦合聚类
 * G. 层列表
 * H. 节点摘要索引
 *
 * 用法: node ua-tour-analyze.js <input.json> <output.json>
 */

const fs = require('fs');

try {
  const inputPath = process.argv[2];
  const outputPath = process.argv[3];

  if (!inputPath || !outputPath) {
    console.error('用法: node ua-tour-analyze.js <input.json> <output.json>');
    process.exit(1);
  }

  const raw = fs.readFileSync(inputPath, 'utf8');
  const data = JSON.parse(raw);

  const { nodes, edges, layers } = data;

  // ---------- 辅助数据结构 ----------

  // nodeMap: id -> node
  const nodeMap = new Map();
  for (const n of nodes) {
    nodeMap.set(n.id, n);
  }

  // ---------- A. Fan-In 排名 ----------
  const fanIn = new Map();
  for (const e of edges) {
    fanIn.set(e.target, (fanIn.get(e.target) || 0) + 1);
  }

  const fanInRanking = [...fanIn.entries()]
    .map(([id, count]) => ({
      id,
      fanIn: count,
      name: nodeMap.get(id)?.name || id,
    }))
    .sort((a, b) => b.fanIn - a.fanIn)
    .slice(0, 20);

  // ---------- B. Fan-Out 排名 ----------
  const fanOut = new Map();
  for (const e of edges) {
    fanOut.set(e.source, (fanOut.get(e.source) || 0) + 1);
  }

  const fanOutRanking = [...fanOut.entries()]
    .map(([id, count]) => ({
      id,
      fanOut: count,
      name: nodeMap.get(id)?.name || id,
    }))
    .sort((a, b) => b.fanOut - a.fanOut)
    .slice(0, 20);

  // ---------- C. 入口点候选 ----------
  const entryPointFileNames = new Set([
    'index.ts', 'index.js', 'main.ts', 'main.js', 'app.ts', 'app.js',
    'server.ts', 'server.js', 'mod.rs', 'main.go', 'main.py', 'main.rs',
    'manage.py', 'app.py', 'wsgi.py', 'asgi.py', 'run.py', '__main__.py',
    'Application.java', 'Main.java', 'Program.cs', 'config.ru', 'index.php',
    'App.swift', 'Application.kt', 'main.cpp', 'main.c',
  ]);

  // Top 10% fan-out threshold
  const allFanOut = [...fanOut.values()].sort((a, b) => b - a);
  const topPct10Index = Math.max(0, Math.floor(allFanOut.length * 0.1));
  const topFanOutThreshold = allFanOut[topPct10Index] || 0;

  // Bottom 25% fan-in threshold
  const allFanIn = [...fanIn.values()].sort((a, b) => a - b);
  const botPct25Index = Math.min(allFanIn.length - 1, Math.floor(allFanIn.length * 0.75));
  const botFanInThreshold = allFanIn[botPct25Index] || 0;

  const entryPointCandidates = nodes.map((n) => {
    let score = 0;
    const fo = fanOut.get(n.id) || 0;
    const fi = fanIn.get(n.id) || 0;

    if (n.type === 'document') {
      // README.md at project root -> +5
      if (n.name === 'README.md' && n.filePath && !n.filePath.includes('/') && !n.filePath.includes('\\')) {
        score += 5;
      } else if (n.name === 'README.md') {
        // README.md at any location
        score += 5;
      } else if (n.filePath && n.name && n.name.endsWith('.md') && !n.filePath.includes('/') && !n.filePath.includes('\\')) {
        score += 2;
      }
    }

    if (n.type === 'file') {
      // Filename match
      if (entryPointFileNames.has(n.name)) {
        score += 3;
      }
      // At root or one level deep
      if (n.filePath) {
        const parts = n.filePath.replace(/\\/g, '/').split('/');
        if (parts.length <= 2) {
          score += 1;
        }
      }
      // High fan-out (top 10%)
      if (fo >= topFanOutThreshold && fo > 0) {
        score += 1;
      }
      // Low fan-in (bottom 25%)
      if (fi <= botFanInThreshold) {
        score += 1;
      }
    }

    return {
      id: n.id,
      score,
      name: n.name,
      summary: n.summary || '',
      type: n.type,
    };
  }).filter((n) => n.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 10);

  // ---------- D. BFS 依赖链遍历 ----------
  // Find top CODE entry point (skip document nodes for BFS)
  const codeEntryCandidate = entryPointCandidates.find((c) => c.type === 'file') ||
    nodes.find((n) => n.type === 'file' && (n.name === '__init__.py' && n.filePath === 'QuantStudio/__init__.py'));

  let bfsResult = { startNode: '', order: [], depthMap: {}, byDepth: {} };

  if (codeEntryCandidate) {
    const startNode = codeEntryCandidate.id;
    // BFS following imports, calls, exports, depends_on edges
    const forwardEdgeTypes = new Set(['imports', 'calls', 'exports', 'depends_on', 'contains']);
    const adj = new Map();
    for (const e of edges) {
      if (forwardEdgeTypes.has(e.type)) {
        if (!adj.has(e.source)) adj.set(e.source, new Set());
        adj.get(e.source).add(e.target);
      }
    }

    const visited = new Set([startNode]);
    const queue = [{ id: startNode, depth: 0 }];
    const order = [startNode];
    const depthMap = { [startNode]: 0 };
    const byDepth = { '0': [startNode] };

    while (queue.length > 0) {
      const { id, depth } = queue.shift();
      const neighbors = adj.get(id) || new Set();
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          visited.add(neighbor);
          const nd = depth + 1;
          order.push(neighbor);
          depthMap[neighbor] = nd;
          if (!byDepth[nd]) byDepth[nd] = [];
          byDepth[nd].push(neighbor);
          queue.push({ id: neighbor, depth: nd });
        }
      }
    }

    bfsResult = { startNode, order, depthMap, byDepth };
  }

  // ---------- E. 非代码文件清单 ----------
  const nonCodeFiles = {
    documentation: [],
    infrastructure: [],
    data: [],
    config: [],
  };

  for (const n of nodes) {
    const entry = { id: n.id, name: n.name, summary: n.summary || '', type: n.type };
    switch (n.type) {
      case 'document':
        nonCodeFiles.documentation.push(entry);
        break;
      case 'service':
      case 'pipeline':
      case 'resource':
        nonCodeFiles.infrastructure.push(entry);
        break;
      case 'table':
      case 'schema':
      case 'endpoint':
        nonCodeFiles.data.push(entry);
        break;
      case 'config':
        nonCodeFiles.config.push(entry);
        break;
      // 'file' type - skip
    }
  }

  // ---------- F. 紧密耦合聚类 ----------
  // Find bidirectional edges
  const edgeSet = new Set();
  for (const e of edges) {
    edgeSet.add(`${e.source}|||${e.target}`);
  }

  // Build clusters from bidirectional pairs
  const bidirPairs = [];
  for (const e of edges) {
    if (edgeSet.has(`${e.target}|||${e.source}`) && e.source < e.target) {
      bidirPairs.push([e.source, e.target]);
    }
  }

  // Union-Find for initial clusters
  const parent = new Map();
  function find(x) {
    if (!parent.has(x)) parent.set(x, x);
    if (parent.get(x) !== x) parent.set(x, find(parent.get(x)));
    return parent.get(x);
  }
  function union(a, b) {
    const ra = find(a), rb = find(b);
    if (ra !== rb) parent.set(ra, rb);
  }

  for (const [a, b] of bidirPairs) {
    union(a, b);
  }

  // Group by root
  const clusterMap = new Map();
  for (const n of nodes) {
    if (parent.has(n.id)) {
      const r = find(n.id);
      if (!clusterMap.has(r)) clusterMap.set(r, []);
      clusterMap.get(r).push(n.id);
    }
  }

  // Expand clusters: add nodes that connect to 2+ existing members
  const clusters = [...clusterMap.values()].filter((c) => c.length >= 2 && c.length <= 5);

  // Try expanding
  for (const cluster of clusters) {
    const memberSet = new Set(cluster);
    for (const n of nodes) {
      if (memberSet.has(n.id)) continue;
      let connections = 0;
      for (const e of edges) {
        if ((e.source === n.id && memberSet.has(e.target)) || (e.target === n.id && memberSet.has(e.source))) {
          connections++;
        }
        if (connections >= 2) break;
      }
      if (connections >= 2 && cluster.length < 5) {
        cluster.push(n.id);
        memberSet.add(n.id);
      }
    }
  }

  // Count edges within each cluster
  const clusterResults = clusters
    .map((c) => {
      const memberSet = new Set(c);
      let edgeCount = 0;
      for (const e of edges) {
        if (memberSet.has(e.source) && memberSet.has(e.target)) {
          edgeCount++;
        }
      }
      return { nodes: c, edgeCount };
    })
    .sort((a, b) => b.edgeCount - a.edgeCount)
    .slice(0, 10);

  // ---------- G. 层列表 ----------
  const layerList = layers || [];
  const layersResult = {
    count: layerList.length,
    list: layerList.map((l) => ({
      id: l.id,
      name: l.name,
      description: l.description || '',
    })),
  };

  // ---------- H. 节点摘要索引 ----------
  const nodeSummaryIndex = {};
  for (const n of nodes) {
    nodeSummaryIndex[n.id] = {
      name: n.name,
      type: n.type,
      summary: n.summary || '',
    };
  }

  // ---------- 输出结果 ----------
  const result = {
    scriptCompleted: true,
    entryPointCandidates,
    fanInRanking,
    fanOutRanking,
    bfsTraversal: bfsResult,
    nonCodeFiles,
    clusters: clusterResults,
    layers: layersResult,
    nodeSummaryIndex,
    totalNodes: nodes.length,
    totalEdges: edges.length,
  };

  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2), 'utf8');
  console.log(`分析完成。共 ${nodes.length} 个节点, ${edges.length} 条边。`);
  console.log(`结果已写入: ${outputPath}`);
  process.exit(0);

} catch (err) {
  console.error('分析脚本执行失败:', err.message);
  console.error(err.stack);
  process.exit(1);
}
