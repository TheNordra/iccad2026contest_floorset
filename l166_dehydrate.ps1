$root = "C:\Users\.01\OneDrive"
$before = (Get-PSDrive C).Free
"start free: {0:N2} GB  {1}" -f ($before/1GB), (Get-Date -f "HH:mm:ss")
# +U marks online-only (dehydrate), -P clears "always keep on this device".
# Reversible: opening a file re-downloads it. Not a delete.
attrib +U -P "$root\*" /s /d 2>&1 | Select-Object -First 20
$after = (Get-PSDrive C).Free
"done  free: {0:N2} GB   reclaimed {1:N2} GB   {2}" -f ($after/1GB), (($after-$before)/1GB), (Get-Date -f "HH:mm:ss")
