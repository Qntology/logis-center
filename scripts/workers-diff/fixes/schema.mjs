// 스키마(workers/migrations/0001)에 없는 열을 쓰던 문장 — D1 배치는 원자적이라 한 문장이 틀리면 그 지역의 문장을 모두 잃고 418 이 났다
export default [
  {
    id: "fix-schema-sales-updated-at", worker: "proxy",
    find: "UPDATE sales SET updated_at = ?, status = ?, tracking = ? WHERE id = ? `).bind( now, item.status, item.index, _order.id",
    replace: "UPDATE sales SET status = ?, tracking = ? WHERE id = ?\n`).bind(\nitem.status, item.index, _order.id",
  },
  {
    id: "fix-schema-event-phone", worker: "proxy", count: 2,
    find: '"event", "number", "address"',
    replace: '"event", "phone", "address"',
  },
  {
    id: "fix-schema-event-phone", worker: "proxy", count: 2,
    find: '"event" = EXCLUDED."event", "number" = EXCLUDED."number", "address" = EXCLUDED."address"',
    replace: '"event" = EXCLUDED."event",\n"phone" = EXCLUDED."phone",\n"address" = EXCLUDED."address"',
  },
  {
    id: "fix-schema-event-phone", worker: "proxy",
    find: 'edge.number ? edge.number : "", edge.address ? edge.address : ""',
    replace: 'edge.phone ? edge.phone : (edge.number ? edge.number : ""),\nedge.address ? edge.address : ""',
  },
  {
    id: "fix-schema-event-phone", worker: "proxy",
    find: 'item.number ? item.number : "", item.address ? item.address : ""',
    replace: 'item.phone ? item.phone : (item.number ? item.number : ""),\nitem.address ? item.address : ""',
  },
];
