import { t } from "./i18n";
import { toolLabel } from "./i18n/labels";
import type { TurnView } from "./types";

/**
 * Human-readable activity label for a running turn.
 *
 * The backend emits tool_call_start while the model streams tool arguments,
 * message_end once the model reply is complete, and tool_result after the
 * tool finishes. Combining the last message status with the tool status gives
 * the current phase without adding new server events.
 */
export function turnPendingLabel(turn: TurnView): string {
  if (turn.status !== "running") return "";
  if (turn.retry) {
    return t("turnPhase.retrying", { attempt: turn.retry.attempt });
  }

  const last = turn.messages[turn.messages.length - 1];
  const runningTool = last?.tools.find((tool) => tool.status === "running");

  if (runningTool) {
    if (last?.status === "streaming") return t("turnPhase.preparingTool");
    if (runningTool.name === "apply_action") return t("turnPhase.submitting");
    return t("turnPhase.runningTool", { tool: toolLabel(runningTool.name) });
  }

  if (!last || last.status === "streaming") return t("turnPhase.thinking");
  return t("turnPhase.continuing");
}
