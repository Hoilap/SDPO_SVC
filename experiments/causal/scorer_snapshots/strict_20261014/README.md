# Strict scorer snapshot, 2026-10-14

Copied before adding the opt-in lenient evaluation scorer. `manifest.json` records
SHA-256 for every original feedback scorer. Original scorer files stay unchanged;
the reevaluation job checks both originals and this snapshot before running.

Lenient scoring is an optimistic diagnostic: any complete boxed math answer
matching the reference, or any explicitly tagged/labelled correct choice, counts.
Repeated text and an unfinished trailing answer do not invalidate an earlier
correct answer. Intermediate numbers and merely mentioned option letters do not
count. Conflicting explicit answers may pass. Code and ToolUse keep original
execution/structure checks. Original format flags, strict scores and the rescue
flag are preserved per response. Training with the new scorer is rejected.

Science GRPO-full and GRPO-no-tail are regenerated on the entire test set using
the existing full-test settings (8 responses/question). Results go to a fresh
job-specific directory; old models and metrics are not overwritten.
