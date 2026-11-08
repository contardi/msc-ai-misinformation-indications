import { useT } from '../i18n'
interface GaugeProps {
  value: number
}

const TICKS = [0, 25, 50, 75, 100]

export default function Gauge({ value }: GaugeProps) {
  const t = useT()
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)))

  return (
    <div
      role="progressbar"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={t.score.gaugeLabel}
      className="select-none"
    >
      <div className="relative h-3 w-full rounded-full bg-linear-to-r from-meter-low via-meter-mid to-meter-high">
        <div
          className="absolute top-1/2 h-6 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-ink shadow-[0_0_0_3px_var(--color-panel)] transition-[left] duration-500 ease-out"
          style={{ left: `${pct}%` }}
        />
      </div>

      <div className="relative mt-2 h-4">
        {TICKS.map((t) => (
          <div
            key={t}
            className="absolute -translate-x-1/2 font-mono text-[10px] text-muted"
            style={{ left: `${t}%` }}
          >
            {t}
          </div>
        ))}
      </div>

      <div className="mt-1 flex justify-between font-mono text-[10px] uppercase tracking-wider text-muted">
        <span>{t.score.gaugeLow}</span>
        <span>{t.score.gaugeHigh}</span>
      </div>
    </div>
  )
}
