# bridge-agent.ps1 - two-way sync guest agent for ntfs-ext4-bridge
#
# Long-polls the bridge's control endpoint and executes namespace operations
# natively on the bridge volume, so Windows' NTFS caches stay coherent while
# the bridge maps the resulting clusters to ext4 source files.
#
# Runs as SYSTEM (scheduled task, see install-agent.ps1). Requires admin for
# fsutil setvaliddata (SeManageVolumePrivilege) and Set-Disk.
#
# Config: C:\ProgramData\BridgeAgent\config.json
#   { "control_url": "http://192.168.122.1:10810", "token": "..." }

$ErrorActionPreference = 'Stop'
$ConfigDir = 'C:\ProgramData\BridgeAgent'
$Config = Get-Content (Join-Path $ConfigDir 'config.json') | ConvertFrom-Json
$CursorFile = Join-Path $ConfigDir 'cursor.json'
$LogFile = Join-Path $ConfigDir 'agent.log'
$AgentVersion = '1.0'

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Add-Content -Path $LogFile -Value $line
    Write-Host $line
}

function Invoke-Bridge($endpoint, $body, $timeoutSec = 40) {
    $json = $body | ConvertTo-Json -Depth 6 -Compress
    return Invoke-RestMethod -Uri "$($Config.control_url)$endpoint" `
        -Method Post -Body $json -ContentType 'application/json' `
        -Headers @{ 'X-Bridge-Token' = $Config.token } `
        -TimeoutSec $timeoutSec
}

function Get-Cursor {
    try { return ([int64](Get-Content $CursorFile | ConvertFrom-Json).cursor) }
    catch { return 0 }
}

function Set-Cursor([int64]$cursor) {
    @{ cursor = $cursor } | ConvertTo-Json | Set-Content $CursorFile
}

function Find-BridgeVolume($volumeSerial) {
    # Match the NTFS volume serial reported by the bridge to a drive letter.
    foreach ($v in Get-CimInstance Win32_Volume |
             Where-Object { $_.DriveLetter -and $_.FileSystem -eq 'NTFS' }) {
        # Win32_Volume SerialNumber is the 32-bit serial; the bridge sends
        # the full 64-bit one. Compare the low 32 bits.
        $low32 = [uint32]("0x" + $volumeSerial.Substring(8, 8))
        if ([uint32]$v.SerialNumber -eq $low32) {
            return $v.DriveLetter.TrimEnd(':')
        }
    }
    return $null
}

function Get-BridgeDiskNumber($driveLetter) {
    return (Get-Partition -DriveLetter $driveLetter).DiskNumber
}

function Resolve-OpPath($driveLetter, $relPath) {
    return "${driveLetter}:\$relPath"
}

function Set-MtimeMs($path, $mtimeMs) {
    if ($mtimeMs) {
        $t = [DateTimeOffset]::FromUnixTimeMilliseconds($mtimeMs).UtcDateTime
        [System.IO.File]::SetLastWriteTimeUtc($path, $t)
    }
}

function Execute-Op($op, $driveLetter) {
    $path = if ($op.path) { Resolve-OpPath $driveLetter $op.path } else { $null }
    switch ($op.op) {
        'mkdir' {
            New-Item -ItemType Directory -Force -Path $path | Out-Null
        }
        'rm' {
            if (Test-Path -LiteralPath $path) {
                Remove-Item -LiteralPath $path -Force -Recurse:([bool]$op.recurse)
            }
        }
        'mv' {
            $dst = Resolve-OpPath $driveLetter $op.dst
            if (-not (Test-Path -LiteralPath $path)) {
                if (Test-Path -LiteralPath $dst) { return }  # already applied
                throw "ENOENT: $path"
            }
            $dstParent = Split-Path $dst -Parent
            if (-not (Test-Path -LiteralPath $dstParent)) {
                New-Item -ItemType Directory -Force -Path $dstParent | Out-Null
            }
            Move-Item -LiteralPath $path -Destination $dst -Force
        }
        'create_sized' {
            $parent = Split-Path $path -Parent
            if (-not (Test-Path -LiteralPath $parent)) {
                New-Item -ItemType Directory -Force -Path $parent | Out-Null
            }
            if (Test-Path -LiteralPath $path) {
                # Fall through to resize semantics
                $fs = [System.IO.File]::Open($path, 'Open', 'ReadWrite')
                $fs.SetLength([int64]$op.size); $fs.Close()
            } elseif ([int64]$op.size -eq 0) {
                New-Item -ItemType File -Path $path | Out-Null
            } else {
                & fsutil file createnew $path $op.size | Out-Null
                if ($LASTEXITCODE -ne 0) { throw "fsutil createnew failed" }
            }
            if ([int64]$op.size -gt 0) {
                # Raise Valid Data Length so reads hit the device (the bridge
                # serves the mapped ext4 bytes) instead of returning zeros.
                & fsutil file setvaliddata $path $op.size | Out-Null
                if ($LASTEXITCODE -ne 0) { throw "fsutil setvaliddata failed" }
            }
            Set-MtimeMs $path $op.mtime_ms
        }
        'resize' {
            if (-not (Test-Path -LiteralPath $path)) {
                # File unknown here yet: degrade to create_sized
                $op.op = 'create_sized'
                Execute-Op $op $driveLetter
                return
            }
            $old = (Get-Item -LiteralPath $path).Length
            $fs = [System.IO.File]::Open($path, 'Open', 'ReadWrite')
            $fs.SetLength([int64]$op.size); $fs.Close()
            if ([int64]$op.size -gt $old -and [int64]$op.size -gt 0) {
                & fsutil file setvaliddata $path $op.size | Out-Null
                if ($LASTEXITCODE -ne 0) { throw "fsutil setvaliddata failed" }
            }
            Set-MtimeMs $path $op.mtime_ms
        }
        'set_mtime' {
            if (Test-Path -LiteralPath $path) { Set-MtimeMs $path $op.mtime_ms }
        }
        'flush_volume' {
            Write-VolumeCache -DriveLetter $driveLetter
        }
        'gate_begin' {
            $diskNum = Get-BridgeDiskNumber $driveLetter
            Write-VolumeCache -DriveLetter $driveLetter
            Set-Disk -Number $diskNum -IsOffline $true
            Log "gate $($op.gate_id): disk $diskNum offline"
            Invoke-Bridge '/v1/gate' @{ gate_id = $op.gate_id; phase = 'offline_confirmed' } | Out-Null
            # Wait for the bridge to finish the offline apply
            while ($true) {
                Start-Sleep -Seconds 2
                $r = Invoke-Bridge '/v1/gate' @{ gate_id = $op.gate_id; phase = 'await_end' }
                if ($r.done) { break }
            }
            Set-Disk -Number $diskNum -IsOffline $false
            Log "gate $($op.gate_id): disk $diskNum online"
            Invoke-Bridge '/v1/gate' @{ gate_id = $op.gate_id; phase = 'online_confirmed' } | Out-Null
        }
        default {
            throw "unknown op: $($op.op)"
        }
    }
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

Log "bridge-agent $AgentVersion starting"
$epoch = $null
$driveLetter = $null

while ($true) {
    try {
        if (-not $epoch) {
            $hello = Invoke-Bridge '/v1/hello' @{
                agent_version = $AgentVersion
                hostname = $env:COMPUTERNAME
            }
            $epoch = $hello.epoch
            $driveLetter = Find-BridgeVolume $hello.volume_serial
            if (-not $driveLetter) {
                Log "bridge volume (serial $($hello.volume_serial)) not found; retrying in 15s"
                $epoch = $null
                Start-Sleep -Seconds 15
                continue
            }
            Log "hello ok: epoch=$epoch volume=${driveLetter}: (serial $($hello.volume_serial))"
        }

        $cursor = Get-Cursor
        $resp = Invoke-Bridge '/v1/poll' @{ cursor = $cursor } 40

        if ($resp.epoch -ne $epoch) {
            # Bridge restarted or a gate completed: our cached NTFS state may
            # be stale. Cycle the disk to drop caches, reset the cursor.
            Log "epoch change $epoch -> $($resp.epoch): cycling disk to drop caches"
            try {
                $diskNum = Get-BridgeDiskNumber $driveLetter
                Set-Disk -Number $diskNum -IsOffline $true
                Start-Sleep -Seconds 2
                Set-Disk -Number $diskNum -IsOffline $false
            } catch {
                Log "disk cycle failed: $_"
            }
            $epoch = $resp.epoch
            Set-Cursor 0
            continue
        }

        if (-not $resp.ops -or $resp.ops.Count -eq 0) { continue }

        $results = @()
        $maxSeq = $cursor
        foreach ($op in $resp.ops) {
            try {
                Execute-Op $op $driveLetter
                $results += @{ seq = $op.seq; status = 'ok' }
            } catch {
                Log "op $($op.seq) ($($op.op) $($op.path)) failed: $_"
                $results += @{ seq = $op.seq; status = 'error'
                               code = 'EFAIL'; message = "$_" }
            }
            $maxSeq = $op.seq
        }

        Invoke-Bridge '/v1/ack' @{ epoch = $epoch; results = $results } | Out-Null
        Set-Cursor $maxSeq
    }
    catch {
        Log "loop error: $_"
        $epoch = $null
        Start-Sleep -Seconds 10
    }
}
