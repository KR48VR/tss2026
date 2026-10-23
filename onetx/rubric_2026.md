RUBRIC_VERSION: 2026.10a

You are an expert rater of what AI systems can do at work, as of October 2026. For each task below, give three ratings. Base them on what frontier AI systems deployed in ordinary workplaces can do today: large language models with file, web and code tools, vision, computer use, and agent frameworks that act inside software. Do not count robots or other physical machines.

exposure: can AI cut the time a skilled worker needs for this task by at least half, with the same quality?
 E0 = No. Even with purpose-built software on top of the model, the time cannot be cut in half.
 E1 = Yes, with a general AI assistant alone (chat with files, web and code), with no custom integration.
 E2 = Yes, but only with AI software built for the purpose or connected to the organisation's own systems and data.

autonomy: could an AI agent do this task end to end today, if it had the usual software access?
 A0 = No. A person must do the core of it.
 A1 = Partly. A person does an essential part (for example anything done in person).
 A2 = Yes, with a person checking or signing off.
 A3 = Yes, without supervision in ordinary use.

barrier: the main reason the task is not A3. One of:
 none (it is A3), physical (needs hands or a body), in_person (must be done face to face), relationship (depends on trust or authority between people), accountability (law, liability or policy requires a responsible person), system_access (needs access to systems or data AI is not usually given), judgement (needs contextual judgement AI still gets wrong), data_absent (the information AI would need is not written down anywhere).

Rate each task as it is typically done, not its easiest version. The occupation in brackets is context only. Reply with JSON only, no prose: a JSON array of objects with keys n, exposure, autonomy, barrier, in the same order as the tasks.
