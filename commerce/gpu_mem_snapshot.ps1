# Per-process dedicated / shared GPU memory (WDDM counters), appended to gpu_mem_snapshot.log
$log = Join-Path $PSScriptRoot 'gpu_mem_snapshot.log'
$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$ded = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples
$sh  = (Get-Counter '\GPU Process Memory(*)\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples
$rows = @{}
foreach ($s in $ded) { if ($s.InstanceName -match 'pid_(\d+)') { $p=[int]$matches[1]; if(-not $rows[$p]){$rows[$p]=@{d=0;s=0}}; $rows[$p].d += $s.CookedValue } }
foreach ($s in $sh)  { if ($s.InstanceName -match 'pid_(\d+)') { $p=[int]$matches[1]; if(-not $rows[$p]){$rows[$p]=@{d=0;s=0}}; $rows[$p].s += $s.CookedValue } }
$out = foreach ($p in $rows.Keys) {
  $pr = Get-Process -Id $p -ErrorAction SilentlyContinue
  [pscustomobject]@{ PID=$p; Name=($(if($pr){$pr.ProcessName}else{'?'})); DedicatedMB=[math]::Round($rows[$p].d/1MB,1); SharedMB=[math]::Round($rows[$p].s/1MB,1) }
}
$total = ($out | Measure-Object DedicatedMB -Sum).Sum
$tag = $args[0]
Add-Content $log "===== $stamp  [$tag]  total dedicated = $([math]::Round($total,1)) MB ====="
$out | Where-Object { $_.DedicatedMB -ge 1 -or $_.SharedMB -ge 50 } | Sort-Object DedicatedMB -Descending | Format-Table -AutoSize | Out-String -Width 200 | Add-Content $log
