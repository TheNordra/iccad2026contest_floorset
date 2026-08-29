Subject: Problem C (cadc1075) — Dependency handling for the Final evaluation environment

Dear Organizers,

While preparing our Final submission, we noticed that two official documents
appear to give inconsistent guidance on dependency handling. We would be
grateful for clarification on the following three points.

**1. Will SciPy be pre-installed in the Final evaluation environment?**

Section 2 of the Beta submission guidelines states:

> "Leave requirements.txt EMPTY (zero bytes). The contest evaluation
> environment already provides: numpy, torch, scipy, numba, tqdm, shapely,
> threadpoolctl..."

Section 2(a) of the Beta evaluation report, however, states:

> "Several submissions import packages such as torch-geometric, torch-scatter,
> and scipy... Do not assume any package beyond the Python standard library is
> available."

Our optimizer uses SciPy, so we would like to confirm whether it will be
pre-installed in the Final evaluation environment. If possible, we would also
appreciate a list of the pre-installed packages and their versions — NumPy and
SciPy in particular — so that we can ensure compatibility.

**2. How is a non-empty requirements.txt installed without network access?**

The system specification states `Internet access: No`. If a participant submits
a non-empty requirements.txt, will the evaluation system run `pip install`
against a local package mirror or a pre-downloaded package cache? Or could a
non-empty requirements.txt instead cause an installation failure because no
network is available?

**3. Is it permitted to include third-party packages in the submission?**

As a contingency, we are considering including the required third-party package
in our submission archive, approximately 120 MB in size. Would this comply with
the submission rules?

The submission guidelines prohibit unused large binaries, with a possible
disqualification. If the bundled package is loaded only when the evaluation
environment does not already provide it — and is not used at all when the
environment does — would it still be considered an "unused large binary"?

We would like to confirm these points in advance so that a difference in
interpretation does not affect our Final submission.

Thank you very much for your time and assistance.

Best regards,
cadc1075
