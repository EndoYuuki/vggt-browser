import { useJobStore } from "../state/jobStore";

export function ModelSelect() {
  const models = useJobStore((s) => s.models);
  const selected = useJobStore((s) => s.selectedModel);
  const selectModel = useJobStore((s) => s.selectModel);

  return (
    <label className="field">
      <span>Model</span>
      <select
        value={selected ?? ""}
        onChange={(e) => selectModel(e.target.value)}
      >
        {models.map((m) => (
          <option key={m.name} value={m.name}>
            {m.display_name}
          </option>
        ))}
      </select>
      {selected && (
        <small>{models.find((m) => m.name === selected)?.description}</small>
      )}
    </label>
  );
}
