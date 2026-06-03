import type { HyperparamSpec } from "@/api/ml";

interface Props {
  spec: Record<string, HyperparamSpec>;
  values: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
}

export default function HyperparamForm({ spec, values, onChange }: Props) {
  const entries = Object.entries(spec);
  if (entries.length === 0) {
    return (
      <p className="text-sm text-slate-500">
        This algorithm has no exposed hyperparameters — sklearn defaults will be used.
      </p>
    );
  }
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {entries.map(([key, s]) => {
        const label = s.label ?? key;
        const v = values[key] ?? s.default;
        const isNoneInt = s.type === "int_or_none";
        return (
          <div key={key}>
            <label className="label">
              {label}
              <span className="ml-2 text-[10px] font-normal text-slate-500">{s.type}</span>
            </label>

            {/* int / int_or_none */}
            {(s.type === "int" || s.type === "int_or_none") && (
              <div className="flex items-center gap-2">
                {isNoneInt && (
                  <label className="flex items-center gap-1 text-xs text-slate-400">
                    <input
                      type="checkbox"
                      checked={v === null || v === undefined}
                      onChange={(e) =>
                        onChange(key, e.target.checked ? null : (s.default ?? s.min ?? 1))
                      }
                    />
                    auto
                  </label>
                )}
                <input
                  type="number"
                  className="input"
                  disabled={isNoneInt && (v === null || v === undefined)}
                  value={
                    v === null || v === undefined
                      ? ""
                      : Number(v as number)
                  }
                  min={s.min ?? undefined}
                  max={s.max ?? undefined}
                  step={s.step ?? 1}
                  onChange={(e) => {
                    const n = e.target.value;
                    onChange(key, n === "" ? null : Number(n));
                  }}
                />
              </div>
            )}

            {/* float */}
            {s.type === "float" && (
              <input
                type="number"
                className="input"
                value={v === null || v === undefined ? "" : Number(v as number)}
                min={s.min ?? undefined}
                max={s.max ?? undefined}
                step={s.step ?? 0.1}
                onChange={(e) =>
                  onChange(key, e.target.value === "" ? null : parseFloat(e.target.value))
                }
              />
            )}

            {/* select */}
            {s.type === "select" && (
              <select
                className="input"
                value={String(v ?? s.default ?? "")}
                onChange={(e) => {
                  const raw = e.target.value;
                  const match = (s.options ?? []).find((o) => String(o) === raw);
                  onChange(key, match ?? raw);
                }}
              >
                {(s.options ?? []).map((o, i) => (
                  <option key={i} value={String(o)}>
                    {String(o)}
                  </option>
                ))}
              </select>
            )}

            {/* csv_int — comma-separated ints */}
            {s.type === "csv_int" && (
              <input
                type="text"
                className="input"
                value={
                  Array.isArray(v)
                    ? (v as number[]).join(",")
                    : v === null || v === undefined
                    ? String(s.default ?? "")
                    : String(v)
                }
                placeholder="e.g. 128,64"
                onChange={(e) => onChange(key, e.target.value)}
              />
            )}

            {/* bool */}
            {s.type === "bool" && (
              <label className="flex items-center gap-2 text-sm text-slate-200">
                <input
                  type="checkbox"
                  checked={Boolean(v)}
                  onChange={(e) => onChange(key, e.target.checked)}
                />
                {label}
              </label>
            )}
          </div>
        );
      })}
    </div>
  );
}
