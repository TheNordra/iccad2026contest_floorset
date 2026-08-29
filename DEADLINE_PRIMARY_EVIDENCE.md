# Deadline — primary evidence (answers HackMD review A)

> 🗓️ **Problem C Final deadline = 2026-08-31 23:59 GMT+8.**
> Confirmed 2026-08-28 from the organiser's own text. Do not re-derive this.

## The conflict, and why both documents were right

Two organiser communications say different things, and neither is wrong:

| source | says | scope |
|---|---|---|
| `Final Submission Guidelines_ABC.pdf` (in `~/Downloads`, downloaded 2026-08-25 18:52) | "The deadline for Final submission is postponed to August 28, 2026 (17:00, GMT+8)." | **all problems A/B/C** |
| organiser clarification mail, quoted below | Aug 31 23:59 GMT+8 | **Problem C only** |

The clarification mail, verbatim as supplied by the user 2026-08-28:

> We would like to clarify the deadline extension announced in our previous
> email. Please note that the extension to August 31, 2026, 23:59 (GMT+8)
> applies only to Problem C. If your team is participating in any other problem,
> the original Final Submission deadline remains unchanged: August 28, 2026,
> 17:00 (GMT+8).

**We are Problem C.** `README.md:5` ("ICCAD 2026 CAD Contest Problem C, The
FloorSet Challenge"), team `cadc1075` (`make_submission.py:61`), and the archive
is `cadc<team_id>.tar.gz` under the Problem B/C Google Drive route. So the
extension applies to us: **2026-08-31 23:59 GMT+8**.

## 🔑 The reusable lesson

**The extension is PROBLEM-SCOPED, and neither document says so on its own.**
The ABC-wide PDF is problem-agnostic and reads as authoritative; the mail that
narrows it is a separate message. A session holding only the PDF concludes
08-28 and is behaving correctly on its evidence. A session holding only the mail
concludes 08-31 and is also behaving correctly.

This is the same failure shape the ledger already records under "primary text is
the authority" — but with a twist worth keeping: **being the primary text is not
enough; a primary text can be superseded for YOUR scope by a later one that
never mentions your scope by name.** When two sources conflict, check whether
they are quantified over the same set before deciding which is stale.

## What was stale, and what was corrected

At the moment this was resolved (2026-08-28 12:24 GMT+8) the tree said 08-28 in:

* `CLAUDE.md` lines 23, 102, 231 — **corrected**
* `VERIFY_RUNBOOK_2026-08-27.md` — no deadline line, nothing to fix
* `HANDOFF_2026-08-27.md` lines 3, 460 — left as-is (dated record of what was
  believed then; this file supersedes it)
* `HANDOFF_2026-08-29_MERGED.md:7` already said 08-31, citing the organiser mail
  of 2026-08-27 — that was right, and is now backed by the text above rather
  than by a citation alone.

## Consequence for shipping

At the time of writing there were **~3 days 11 hours** left, not the ~4.5 hours
the 08-28 17:00 reading implied. Concretely this is what changes:

* the "combine it into Sunday's package" plan in `handover/2026-08-28-drop-absolute-path`
  is **live**, not dead. There is time for a full re-stage + re-verify + re-upload.
* `build_submission.D/cadc1075.tar.gz` is already uploaded and verified, so the
  downside of the extra time is bounded: if anything fails, D stands.
