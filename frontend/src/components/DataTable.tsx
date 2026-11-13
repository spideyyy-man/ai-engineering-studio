interface Props {
  columns: { key: string; label?: string }[];
  rows: Record<string, unknown>[];
  maxHeight?: number;
}

export default function DataTable({ columns, rows, maxHeight = 320 }: Props) {
  return (
    <div
      className="overflow-auto rounded-lg border border-bg-border bg-bg-panel"
      style={{ maxHeight }}
    >
      <table className="min-w-full text-sm">
        <thead className="sticky top-0 bg-bg-card text-xs text-slate-400">
          <tr>
            {columns.map((c) => (
              <th key={c.key} className="whitespace-nowrap px-3 py-2 text-left font-medium">
                {c.label ?? c.key}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="px-3 py-6 text-center text-slate-500">
                No rows
              </td>
            </tr>
          ) : (
            rows.map((r, i) => (
              <tr
                key={i}
                className="border-t border-bg-border odd:bg-bg-panel even:bg-bg-card/40"
              >
                {columns.map((c) => (
                  <td key={c.key} className="whitespace-nowrap px-3 py-1.5 text-slate-200">
                    {formatCell(r[c.key])}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

function formatCell(v: unknown): string {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return String(v);
    return Number.isInteger(v) ? v.toString() : v.toFixed(4);
  }
  return String(v);
}
