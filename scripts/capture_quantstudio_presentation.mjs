/** 为系统介绍截取真实 QSWeb 编辑器；接口使用演示夹具，不访问业务数据。
 * 用法：node scripts/capture_quantstudio_presentation.mjs
 * 前提：同级 QSWeb 的图编辑器测试服务已在 23117 端口启动。
 */
import { chromium } from '../../QSWeb/frontend/node_modules/playwright-core/index.mjs'
import fs from 'node:fs'
import path from 'node:path'
const out = path.resolve('docs/presentation/assets')
fs.mkdirSync(out, { recursive: true })
const spec = [
  ['price', '行情与价格', [], [50, 150]],
  ['momentum', '时序动量', ['price'], [390, 100]],
  ['score', '截面标准化', ['momentum'], [730, 100]],
  ['account', '策略账户', ['score', 'price'], [1070, 160]],
  ['ic', 'IC 分析', ['score', 'price'], [1070, 440]],
  ['report', '综合研究报告', ['account', 'ic'], [1410, 260]],
]
const definitions = spec.map(([id, title, inputs]) => ({
  definition: { id, version: 1 }, title, description: '系统介绍演示组件',
  source_path: `research/${id}.py`, source_category: '研究组件', operations: ['build'], args: [],
  inputs: inputs.map(name => ({ name, title: ({price:'价格',momentum:'动量',score:'因子得分',account:'账户统计',ic:'IC 结果'})[name], type: { kind: 'node' } })),
  outputs: { result: { kind: 'node' } }, args_schema: { type: 'object', properties: {} }, result_args_schema: null,
}))
const document = { schema_version: 1,
  blocks: spec.map(([id, title, inputs]) => ({ id, definition: { id, version: 1 }, operation: 'build',
    args: { Name: title }, inputs: Object.fromEntries(inputs.map(name => [name, { kind: 'reference', block: name, port: 'result' }])) })),
  outputs: { '研究报告': { kind: 'reference', block: 'report', port: 'result' } },
  extensions: { qsweb: { layout: { positions: Object.fromEntries(spec.map(([id,,,p]) => [id, { x:p[0], y:p[1] }])), collapsed: [], viewport: { x:35, y:70, zoom:0.86 } } } },
}
const cache = path.join(process.env.LOCALAPPDATA, 'ms-playwright')
const installed = fs.readdirSync(cache).filter(name => /^chromium-\d+$/.test(name))
  .sort((a, b) => Number(b.split('-')[1]) - Number(a.split('-')[1]))
const executablePath = process.env.GRAPH_TEST_CHROMIUM || installed
  .map(name => path.join(cache, name, 'chrome-win64/chrome.exe')).find(name => fs.existsSync(name))
const browser = await chromium.launch({ headless: true, executablePath })
try {
  const page = await browser.newPage({ viewport: { width: 1720, height: 940 }, deviceScaleFactor: 1.5 })
  await page.addInitScript(() => {
    localStorage.setItem('qsweb_access_token', 'presentation-fixture')
    localStorage.setItem('qsweb_user', JSON.stringify({id:'presentation',username:'研究演示'}))
  })
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url()), p = url.pathname
    let body = []
    if(p.endsWith('/definitions')) body = definitions
    else if(p.endsWith('/run-config')) body = { section_sources: [], lookback_days: 365, trading_day_available: false }
    else if(p.endsWith('/validate')) body = { ...route.request().postDataJSON(), diagnostics: [], descriptions: {} }
    else if(p === '/api/config/pages') body = {enabled_pages:['graph','data','risk','portfolio','report','ai','mining','notebook']}
    else if(p === '/api/config') body = { engine: { type:'CalcEngine', params:{} }, use_temp_cache:true }
    else if(p.includes('/config/preferences')) body = {}
    await route.fulfill({json:body})
  })
  await page.goto('http://127.0.0.1:23117/tests/graph_editor.html?layout')
  await page.waitForFunction(() => window.graphEditorStore?.getState().definitions.length > 0)
  await page.evaluate(({document,definitions}) => {
    window.graphEditorStore.getState().load({id:'demo', revision:1,name:'多因子研究 · 从数据到报告',document})
    window.graphEditorStore.setState({ definitions })
  }, {document,definitions})
  await page.waitForTimeout(1800)
  await page.screenshot({path:path.join(out,'qsweb_graph.png')})
  fs.writeFileSync(path.join(out,'qsweb_graph_fixture.json'),JSON.stringify({definitions,document},null,2))
  console.log('已截取真实 QSWeb 编辑器；图与接口响应为演示夹具。')
} finally { await browser.close() }
