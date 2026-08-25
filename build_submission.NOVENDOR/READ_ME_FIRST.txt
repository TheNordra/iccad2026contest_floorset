DUPLICATE. Do not upload from this directory.

This is byte-identical in content to ../build_submission.D/ (same six graded
files, same 8 tar members, no vendor/). The tars differ by one byte, which is
gzip's embedded mtime, not content.

UPLOAD ../build_submission.D/cadc1075.tar.gz instead. That is the copy whose
own tar was Linux-verified under the official command (l244b_wsl.log, and
l246_wsl.log for the full five-lane chain).

Kept rather than deleted so nothing that references it breaks. If you are
regenerating, l245_novendor.sh now writes to build_submission.D by default.
