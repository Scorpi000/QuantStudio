/**
 * QuantStudio 架构分析脚本
 *
 * 读取 file-nodes、import-edges 和 all-edges 数据,
 * 计算目录分组、节点类型分组、导入邻接矩阵、跨类别依赖、
 * 组间导入频率、组内密度、目录模式匹配、部署拓扑、数据管线、
 * 文档覆盖率和依赖方向等结构化指标。
 *
 * 用法: node ua-arch-analyze.js <input.json> <output.json>
 */

const fs = require('fs');
const path = require('path');

try {
  const inputPath = process.argv[2];
  const outputPath = process.argv[3];

  if (!inputPath || !outputPath) {
    console.error('Usage: node ua-arch-analyze.js <input.json> <output.json>');
    process.exit(1);
  }

  const input = JSON.parse(fs.readFileSync(inputPath, 'utf-8'));
  const { fileNodes, importEdges, allEdges } = input;

  // --- A. Directory Grouping ---
  function getTopGroup(filePath) {
    const parts = filePath.replace(/\\/g, '/').split('/');
    // Common prefix is "QuantStudio" for code, or root-level files
    if (parts.length === 1) return '(root)';
    if (parts[0] === 'QuantStudio') {
      if (parts.length === 2) return 'QuantStudio-root';
      return parts[1]; // e.g., Core, Factor, Tools, etc.
    }
    // docs, tests, examples, scripts, etc.
    return parts[0];
  }

  const directoryGroups = {};
  fileNodes.forEach(node => {
    const group = getTopGroup(node.filePath);
    if (!directoryGroups[group]) directoryGroups[group] = [];
    directoryGroups[group].push(node.id);
  });

  // --- B. Node Type Grouping ---
  const nodeTypeGroups = {};
  fileNodes.forEach(node => {
    if (!nodeTypeGroups[node.type]) nodeTypeGroups[node.type] = [];
    nodeTypeGroups[node.type].push(node.id);
  });

  // --- Build file lookup ---
  const nodeById = {};
  fileNodes.forEach(n => { nodeById[n.id] = n; });

  // --- C. Import Adjacency ---
  const fanIn = {};
  const fanOut = {};

  importEdges.forEach(e => {
    if (!fanOut[e.source]) fanOut[e.source] = new Set();
    if (!fanIn[e.target]) fanIn[e.target] = new Set();
    fanOut[e.source].add(e.target);
    fanIn[e.target].add(e.source);
  });

  const fileFanIn = {};
  const fileFanOut = {};
  Object.entries(fanIn).forEach(([k, v]) => { fileFanIn[k] = v.size; });
  Object.entries(fanOut).forEach(([k, v]) => { fileFanOut[k] = v.size; });

  // --- E. Inter-Group Import Frequency ---
  const interGroup = {};
  importEdges.forEach(e => {
    const srcNode = nodeById[e.source];
    const tgtNode = nodeById[e.target];
    if (!srcNode || !tgtNode) return;
    const srcGroup = getTopGroup(srcNode.filePath);
    const tgtGroup = getTopGroup(tgtNode.filePath);
    if (srcGroup === tgtGroup) return;
    const key = `${srcGroup} -> ${tgtGroup}`;
    interGroup[key] = (interGroup[key] || 0) + 1;
  });

  const interGroupImports = Object.entries(interGroup)
    .map(([k, count]) => {
      const [from, to] = k.split(' -> ');
      return { from, to, count };
    })
    .sort((a, b) => b.count - a.count);

  // --- F. Intra-Group Import Density ---
  const intraGroupDensity = {};
  Object.keys(directoryGroups).forEach(group => {
    intraGroupDensity[group] = { internalEdges: 0, totalEdges: 0, density: 0 };
  });

  importEdges.forEach(e => {
    const srcNode = nodeById[e.source];
    const tgtNode = nodeById[e.target];
    if (!srcNode || !tgtNode) return;
    const srcGroup = getTopGroup(srcNode.filePath);
    const tgtGroup = getTopGroup(tgtNode.filePath);
    if (intraGroupDensity[srcGroup]) intraGroupDensity[srcGroup].totalEdges++;
    if (intraGroupDensity[tgtGroup]) intraGroupDensity[tgtGroup].totalEdges++;
    if (srcGroup === tgtGroup) {
      intraGroupDensity[srcGroup].internalEdges++;
    }
  });

  Object.entries(intraGroupDensity).forEach(([group, data]) => {
    data.density = data.totalEdges > 0 ? Math.round(data.internalEdges / data.totalEdges * 100) / 100 : 0;
  });

  // --- D. Cross-Category Dependency Analysis ---
  const crossCat = {};
  (allEdges || []).forEach(e => {
    const srcNode = nodeById[e.source];
    const tgtNode = nodeById[e.target];
    if (!srcNode || !tgtNode) return;
    if (srcNode.type === tgtNode.type && srcNode.type === 'file') return;
    const key = `${srcNode.type} -> ${tgtNode.type} (${e.type})`;
    crossCat[key] = (crossCat[key] || 0) + 1;
  });

  const crossCategoryEdges = Object.entries(crossCat).map(([k, count]) => {
    const m = k.match(/^(\S+) -> (\S+) \((.+)\)$/);
    return { fromType: m[1], toType: m[2], edgeType: m[3], count };
  }).sort((a, b) => b.count - a.count);

  // --- G. Directory Pattern Matching ---
  const dirPatternMap = {
    'routes': 'api', 'api': 'api', 'controllers': 'api', 'endpoints': 'api', 'handlers': 'api',
    'services': 'service', 'core': 'service', 'lib': 'service', 'domain': 'service', 'logic': 'service',
    'models': 'data', 'db': 'data', 'data': 'data', 'persistence': 'data', 'repository': 'data', 'entities': 'data',
    'components': 'ui', 'views': 'ui', 'pages': 'ui', 'ui': 'ui', 'layouts': 'ui', 'screens': 'ui',
    'middleware': 'middleware', 'plugins': 'middleware', 'interceptors': 'middleware', 'guards': 'middleware',
    'utils': 'utility', 'helpers': 'utility', 'common': 'utility', 'shared': 'utility', 'tools': 'utility',
    'config': 'config', 'constants': 'config', 'env': 'config', 'settings': 'config',
    '__tests__': 'test', 'test': 'test', 'tests': 'test', 'spec': 'test', 'specs': 'test',
    'types': 'types', 'interfaces': 'types', 'schemas': 'types', 'contracts': 'types', 'dtos': 'types',
    'hooks': 'hooks', 'store': 'state', 'state': 'state', 'reducers': 'state', 'actions': 'state', 'slices': 'state',
    'assets': 'assets', 'static': 'assets', 'public': 'assets',
    'migrations': 'data', 'management': 'config', 'commands': 'config', 'templatetags': 'utility', 'signals': 'service',
    'serializers': 'api', 'cmd': 'entry', 'internal': 'service', 'pkg': 'utility',
    'src/main/java': 'service', 'src/test/java': 'test',
    'dto': 'types', 'request': 'types', 'response': 'types', 'entity': 'data', 'controller': 'api', 'routers': 'api',
    'composables': 'service', 'blueprints': 'api',
    'mailers': 'service', 'jobs': 'service', 'channels': 'service', 'bin': 'entry',
    'docs': 'documentation', 'documentation': 'documentation', 'wiki': 'documentation',
    'deploy': 'infrastructure', 'deployment': 'infrastructure', 'infra': 'infrastructure', 'infrastructure': 'infrastructure',
    '.github': 'ci-cd', '.gitlab': 'ci-cd', '.circleci': 'ci-cd',
    'k8s': 'infrastructure', 'kubernetes': 'infrastructure', 'helm': 'infrastructure', 'charts': 'infrastructure',
    'terraform': 'infrastructure', 'tf': 'infrastructure', 'docker': 'infrastructure',
    'sql': 'data', 'database': 'data', 'schema': 'data',
    'examples': 'documentation', 'scripts': 'utility',
  };

  const patternMatches = {};
  Object.keys(directoryGroups).forEach(group => {
    const lower = group.toLowerCase();
    if (dirPatternMap[lower]) {
      patternMatches[group] = dirPatternMap[lower];
    } else if (lower === 'quantstudio-root') {
      patternMatches[group] = 'entry';
    } else {
      // Try partial matching for QuantStudio-specific dirs
      if (lower.includes('factor')) patternMatches[group] = 'service';
      else if (lower.includes('backtest')) patternMatches[group] = 'service';
      else if (lower.includes('risk')) patternMatches[group] = 'service';
      else if (lower.includes('portfolio')) patternMatches[group] = 'service';
      else if (lower.includes('tool')) patternMatches[group] = 'utility';
      else if (lower.includes('core')) patternMatches[group] = 'service';
      else patternMatches[group] = 'unknown';
    }
  });

  // File-level pattern matches for file nodes
  const filePatternMatches = {};
  fileNodes.forEach(node => {
    const name = node.name;
    const fp = node.filePath;
    if (/\.test\./.test(name) || /\.spec\./.test(name) || /^test_/.test(name) || /_test\./.test(name) || /Test\./.test(name)) {
      filePatternMatches[node.id] = 'test';
    } else if (/\.d\.ts$/.test(name)) {
      filePatternMatches[node.id] = 'types';
    } else if (/^__init__\.py$/.test(name)) {
      filePatternMatches[node.id] = 'entry';
    } else if (/^api\.py$/.test(name)) {
      filePatternMatches[node.id] = 'api';
    } else if (/^setup\.py$/.test(name) || /^pyproject\.toml$/.test(name) || /^Cargo\.toml$/.test(name) || /^go\.mod$/.test(name) || /^Gemfile$/.test(name) || /^pom\.xml$/.test(name) || /^build\.gradle$/.test(name) || /^composer\.json$/.test(name)) {
      filePatternMatches[node.id] = 'config';
    } else if (/^Dockerfile/.test(name) || /^docker-compose/.test(name)) {
      filePatternMatches[node.id] = 'infrastructure';
    } else if (/\.tf$/.test(name) || /\.tfvars$/.test(name)) {
      filePatternMatches[node.id] = 'infrastructure';
    } else if (/\.sql$/.test(name)) {
      filePatternMatches[node.id] = 'data';
    } else if (/\.graphql$/.test(name) || /\.gql$/.test(name) || /\.proto$/.test(name)) {
      filePatternMatches[node.id] = 'types';
    } else if (/\.md$/.test(name) || /\.rst$/.test(name)) {
      filePatternMatches[node.id] = 'documentation';
    } else if (/^Makefile$/.test(name)) {
      filePatternMatches[node.id] = 'infrastructure';
    }
  });

  // --- H. Deployment Topology Detection ---
  const allFilePaths = fileNodes.map(n => n.filePath);
  const infraFiles = [];

  const hasDockerfile = allFilePaths.some(fp => { const r = /Dockerfile/i.test(fp); if (r) infraFiles.push(fp); return r; });
  const hasCompose = allFilePaths.some(fp => { const r = /docker-compose/i.test(fp); if (r) infraFiles.push(fp); return r; });
  const hasK8s = allFilePaths.some(fp => { const r = /k8s|kubernetes|\.yaml$/i.test(fp) && /manifest|deploy/i.test(fp); if (r) infraFiles.push(fp); return r; });
  const hasTerraform = allFilePaths.some(fp => { const r = /\.tf$|\.tfvars$/i.test(fp); if (r) infraFiles.push(fp); return r; });
  const hasCI = allFilePaths.some(fp => {
    const r = /\.github\/workflows|\.gitlab-ci|Jenkinsfile|\.circleci/i.test(fp);
    if (r) infraFiles.push(fp);
    return r;
  });

  const deploymentTopology = {
    hasDockerfile, hasCompose, hasK8s, hasTerraform, hasCI,
    infraFiles: [...new Set(infraFiles)]
  };

  // --- I. Data Pipeline Detection ---
  const schemaFiles = allFilePaths.filter(fp => /\.(sql|graphql|gql|proto|prisma)$/i.test(fp));
  const migrationFiles = allFilePaths.filter(fp => /migration/i.test(fp));
  const dataModelFiles = allFilePaths.filter(fp => /model/i.test(fp) || /entity/i.test(fp));
  const apiHandlerFiles = allFilePaths.filter(fp => /route|handler|controller|endpoint/i.test(fp));

  const dataPipeline = {
    schemaFiles, migrationFiles, dataModelFiles, apiHandlerFiles
  };

  // --- J. Documentation Coverage ---
  const docGroups = new Set();
  Object.entries(directoryGroups).forEach(([group, ids]) => {
    const hasDoc = ids.some(id => {
      const node = nodeById[id];
      return node && (node.type === 'document' || /\.(md|rst)$/i.test(node.name));
    });
    if (hasDoc) docGroups.add(group);
  });

  const totalGroups = Object.keys(directoryGroups).length;
  const docCoverage = {
    groupsWithDocs: docGroups.size,
    totalGroups,
    coverageRatio: totalGroups > 0 ? Math.round(docGroups.size / totalGroups * 100) / 100 : 0,
    undocumentedGroups: Object.keys(directoryGroups).filter(g => !docGroups.has(g))
  };

  // --- K. Dependency Direction ---
  const groupPairCounts = {};
  importEdges.forEach(e => {
    const srcNode = nodeById[e.source];
    const tgtNode = nodeById[e.target];
    if (!srcNode || !tgtNode) return;
    const srcGroup = getTopGroup(srcNode.filePath);
    const tgtGroup = getTopGroup(tgtNode.filePath);
    if (srcGroup === tgtGroup) return;
    const fwd = `${srcGroup}->${tgtGroup}`;
    const rev = `${tgtGroup}->${srcGroup}`;
    groupPairCounts[fwd] = (groupPairCounts[fwd] || 0) + 1;
  });

  const seen = new Set();
  const dependencyDirection = [];
  Object.entries(groupPairCounts).forEach(([key, count]) => {
    const [a, b] = key.split('->');
    const pairKey = [a, b].sort().join('|');
    if (seen.has(pairKey)) return;
    seen.add(pairKey);
    const fwd = groupPairCounts[`${a}->${b}`] || 0;
    const bwd = groupPairCounts[`${b}->${a}`] || 0;
    if (fwd > bwd) {
      dependencyDirection.push({ dependent: a, dependsOn: b, count: fwd });
    } else if (bwd > fwd) {
      dependencyDirection.push({ dependent: b, dependsOn: a, count: bwd });
    } else if (fwd > 0) {
      dependencyDirection.push({ dependent: a, dependsOn: b, count: fwd });
    }
  });
  dependencyDirection.sort((a, b) => b.count - a.count);

  // --- File Stats ---
  const filesPerGroup = {};
  Object.entries(directoryGroups).forEach(([group, ids]) => {
    filesPerGroup[group] = ids.length;
  });
  const nodeTypeCounts = {};
  Object.entries(nodeTypeGroups).forEach(([type, ids]) => {
    nodeTypeCounts[type] = ids.length;
  });

  // --- Output ---
  const result = {
    scriptCompleted: true,
    directoryGroups,
    nodeTypeGroups,
    crossCategoryEdges,
    interGroupImports,
    intraGroupDensity,
    patternMatches,
    filePatternMatches,
    deploymentTopology,
    dataPipeline,
    docCoverage,
    dependencyDirection,
    fileStats: {
      totalFileNodes: fileNodes.length,
      filesPerGroup,
      nodeTypeCounts
    },
    fileFanIn,
    fileFanOut
  };

  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2), 'utf-8');
  console.log('Analysis complete. Output written to', outputPath);
  process.exit(0);

} catch (err) {
  console.error('Error:', err.message);
  console.error(err.stack);
  process.exit(1);
}
