/**
 * 准备 ua-tour-analyze.js 的输入文件
 * 合并 file-nodes-for-tour.json + all-edges.json + 层级定义
 */
const fs = require('fs');
const path = require('path');

const baseDir = path.resolve(process.argv[2] || '.');
const outputPath = path.resolve(process.argv[3]);

const nodes = JSON.parse(fs.readFileSync(path.join(baseDir, 'file-nodes-for-tour.json'), 'utf8'));
const edges = JSON.parse(fs.readFileSync(path.join(baseDir, 'all-edges.json'), 'utf8'));

// Define layers based on project architecture
const layers = [
  { id: "layer:core", name: "Core", description: "计算图引擎核心：Node、Context、Engine、Panel 等基础设施" },
  { id: "layer:factor", name: "Factor", description: "因子框架：FactorDB、FactorTable、Factor、FactorOperator 等因子计算与管理" },
  { id: "layer:backtest", name: "BackTest", description: "回测框架：BTNode、BTReport、截面因子测试、策略回测、业绩归因" },
  { id: "layer:risk", name: "Risk", description: "风险模型：RiskDB、HDF5RDB、Barra 多因子风险模型" },
  { id: "layer:portfolio", name: "PortfolioConstructor", description: "组合优化：目标函数、约束条件、CVXPY 凸优化求解" },
  { id: "layer:tools", name: "Tools", description: "工具模块：数学、日期时间、数据预处理、文件IO、SQL、可视化等" },
];

const input = { nodes, edges, layers };
fs.writeFileSync(outputPath, JSON.stringify(input), 'utf8');
console.log(`输入文件已写入: ${outputPath} (nodes: ${nodes.length}, edges: ${edges.length}, layers: ${layers.length})`);
