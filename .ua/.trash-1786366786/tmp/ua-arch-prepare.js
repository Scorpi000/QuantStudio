/**
 * 合并 file-nodes、import-edges 和 all-edges 为分析脚本的输入格式
 */
const fs = require('fs');

const fileNodes = JSON.parse(fs.readFileSync('C:/Users/hst/Project/QuantStudio/.ua/tmp/file-nodes-for-arch.json', 'utf-8'));
const importEdges = JSON.parse(fs.readFileSync('C:/Users/hst/Project/QuantStudio/.ua/tmp/import-edges.json', 'utf-8'));
const allEdges = JSON.parse(fs.readFileSync('C:/Users/hst/Project/QuantStudio/.ua/tmp/all-edges.json', 'utf-8'));

const input = { fileNodes, importEdges, allEdges };
fs.writeFileSync('C:/Users/hst/Project/QuantStudio/.ua/tmp/ua-arch-input.json', JSON.stringify(input), 'utf-8');
console.log('Input prepared. Nodes:', fileNodes.length, 'Import edges:', importEdges.length, 'All edges:', allEdges.length);
