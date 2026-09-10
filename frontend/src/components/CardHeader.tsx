export default function CardHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div className="mb-5">
      <h3 className="font-semibold text-[var(--color-ink)]">{title}</h3>
      {subtitle && <p className="text-xs text-[var(--color-ink-dim)] mt-0.5">{subtitle}</p>}
    </div>
  );
}
